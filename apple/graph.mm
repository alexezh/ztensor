// zf::apple::Graph on MPSGraph. See graph.h.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "graph.h"

#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>

namespace zf {
namespace apple {

namespace {

id<MTLDevice> device() {
  static id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  return d;
}

NSArray<NSNumber*>* ns_shape(const Shape& s) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray arrayWithCapacity:s.size()];
  for (int v : s) [a addObject:@(v)];
  return a;
}

std::size_t count(NSArray<NSNumber*>* shape) {
  std::size_t n = 1;
  for (NSNumber* v in shape) n *= static_cast<std::size_t>(v.longLongValue);
  return n;
}

}  // namespace

bool available() { return device() != nil; }

// --- Buffer -------------------------------------------------------------------

struct Buffer::Impl {
  id<MTLBuffer> buffer = nil;
  std::size_t n = 0;
  Type type = Type::kFloat;
  // The tensor data last made for this buffer, kept while the shape stays.
  MPSGraphTensorData* data = nil;
  NSArray<NSNumber*>* data_shape = nil;
};

Buffer::Buffer(std::size_t n, Type type) : impl_(std::make_shared<Impl>()) {
  if (device() == nil) throw std::runtime_error("zf::apple: no Metal device");
  impl_->n = n;
  impl_->type = type;
  // Metal refuses an empty buffer.
  impl_->buffer = [device() newBufferWithLength:(n > 0 ? n : 1) * 4 options:MTLResourceStorageModeShared];
  if (impl_->buffer == nil) throw std::runtime_error("zf::apple: buffer allocation failed");
  std::memset(impl_->buffer.contents, 0, (n > 0 ? n : 1) * 4);
}

std::size_t Buffer::size() const { return impl_ ? impl_->n : 0; }
Buffer::Type Buffer::type() const { return impl_ ? impl_->type : Type::kFloat; }
float* Buffer::data() { return static_cast<float*>(impl_->buffer.contents); }
const float* Buffer::data() const { return static_cast<const float*>(impl_->buffer.contents); }
std::int32_t* Buffer::ints() { return static_cast<std::int32_t*>(impl_->buffer.contents); }
const std::int32_t* Buffer::ints() const { return static_cast<const std::int32_t*>(impl_->buffer.contents); }
void Buffer::zero() { std::memset(impl_->buffer.contents, 0, impl_->n * 4); }

// --- Graph --------------------------------------------------------------------

struct Graph::Impl {
  MPSGraph* g = nil;
  id<MTLCommandQueue> queue = nil;
  NSMutableArray<MPSGraphTensor*>* nodes = nil;

  MPSGraphTensor* at(Node n) const {
    if (n.id < 0 || n.id >= static_cast<int>(nodes.count)) throw std::invalid_argument("zf::apple: bad node");
    return nodes[static_cast<NSUInteger>(n.id)];
  }
  // Each node's inputs, for the gradients: MPSGraph asserts on a tensor
  // the loss does not depend on, so those are found here first.
  std::vector<std::vector<int>> parents;

  Node put(MPSGraphTensor* t, std::initializer_list<Node> from = {}) {
    return put(t, std::vector<Node>(from));
  }
  Node put(MPSGraphTensor* t, const std::vector<Node>& from) {
    [nodes addObject:t];
    std::vector<int> ids;
    for (Node n : from) ids.push_back(n.id);
    parents.push_back(std::move(ids));
    return Node{static_cast<int>(nodes.count) - 1};
  }
  int axis(Node n, int a) const {
    const int rank = static_cast<int>(at(n).shape.count);
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) throw std::invalid_argument("zf::apple: bad axis");
    return a;
  }
  NSArray<NSNumber*>* axes(Node n, const std::vector<int>& as) const {
    NSMutableArray<NSNumber*>* out = [NSMutableArray array];
    for (int a : as) [out addObject:@(axis(n, a))];
    return out;
  }
  MPSGraphTensor* scalar(double v) { return [g constantWithScalar:v dataType:MPSDataTypeFloat32]; }
};

Graph::Graph() : impl_(std::make_unique<Impl>()) {
  if (device() == nil) throw std::runtime_error("zf::apple: no Metal device");
  impl_->g = [MPSGraph new];
  impl_->queue = [device() newCommandQueue];
  impl_->nodes = [NSMutableArray array];
}
Graph::~Graph() = default;
Graph::Graph(Graph&&) noexcept = default;
Graph& Graph::operator=(Graph&&) noexcept = default;

Node Graph::input(const Shape& shape) {
  return impl_->put([impl_->g placeholderWithShape:ns_shape(shape) dataType:MPSDataTypeFloat32 name:nil]);
}
Node Graph::input_ints(const Shape& shape) {
  return impl_->put([impl_->g placeholderWithShape:ns_shape(shape) dataType:MPSDataTypeInt32 name:nil]);
}

Shape Graph::shape(Node n) const {
  Shape s;
  for (NSNumber* v in impl_->at(n).shape) s.push_back(v.intValue);
  return s;
}
std::size_t Graph::elements(Node n) const { return count(impl_->at(n).shape); }

Node Graph::constant(double v) { return impl_->put(impl_->scalar(v)); }
Node Graph::zeros(const Shape& shape) {
  return impl_->put([impl_->g constantWithScalar:0.0 shape:ns_shape(shape) dataType:MPSDataTypeFloat32]);
}

#define ZF_BINARY(op, sel)                                                                    \
  Node Graph::op(Node a, Node b) {                                                            \
    return impl_->put([impl_->g sel##WithPrimaryTensor:impl_->at(a) secondaryTensor:impl_->at(b) name:nil], {a, b}); \
  }
ZF_BINARY(add, addition)
ZF_BINARY(sub, subtraction)
ZF_BINARY(mul, multiplication)
ZF_BINARY(div, division)
ZF_BINARY(maximum, maximum)
ZF_BINARY(minimum, minimum)
#undef ZF_BINARY

#define ZF_UNARY(op, sel) \
  Node Graph::op(Node a) { return impl_->put([impl_->g sel##WithTensor:impl_->at(a) name:nil], {a}); }
ZF_UNARY(neg, negative)
ZF_UNARY(exp, exponent)
ZF_UNARY(log, logarithm)
ZF_UNARY(abs, absolute)
ZF_UNARY(sqrt, squareRoot)
ZF_UNARY(rsqrt, reciprocalSquareRoot)
ZF_UNARY(square, square)
ZF_UNARY(relu, reLU)
ZF_UNARY(sigmoid, sigmoid)
#undef ZF_UNARY

Node Graph::greater(Node a, Node b) {
  MPSGraphTensor* t = [impl_->g greaterThanWithPrimaryTensor:impl_->at(a) secondaryTensor:impl_->at(b) name:nil];
  return impl_->put([impl_->g castTensor:t toType:MPSDataTypeFloat32 name:@"greater"], {a, b});
}

Node Graph::sum(Node a, const std::vector<int>& axes) {
  return impl_->put([impl_->g reductionSumWithTensor:impl_->at(a) axes:impl_->axes(a, axes) name:nil], {a});
}
Node Graph::max(Node a, const std::vector<int>& axes) {
  return impl_->put([impl_->g reductionMaximumWithTensor:impl_->at(a) axes:impl_->axes(a, axes) name:nil], {a});
}
Node Graph::sum_all(Node a) {
  MPSGraphTensor* flat = [impl_->g reshapeTensor:impl_->at(a) withShape:@[ @(-1) ] name:nil];
  return impl_->put([impl_->g reductionSumWithTensor:flat axis:0 name:nil], {a});
}

Node Graph::matmul(Node a, Node b) {
  return impl_->put([impl_->g matrixMultiplicationWithPrimaryTensor:impl_->at(a) secondaryTensor:impl_->at(b) name:nil], {a, b});
}
Node Graph::transpose(Node a, int axis0, int axis1) {
  return impl_->put([impl_->g transposeTensor:impl_->at(a)
                                    dimension:impl_->axis(a, axis0)
                                withDimension:impl_->axis(a, axis1)
                                         name:nil],
                     {a});
}
Node Graph::reshape(Node a, const Shape& shape) {
  return impl_->put([impl_->g reshapeTensor:impl_->at(a) withShape:ns_shape(shape) name:nil], {a});
}
Node Graph::slice(Node a, int axis, int start, int length) {
  return impl_->put([impl_->g sliceTensor:impl_->at(a) dimension:impl_->axis(a, axis) start:start length:length name:nil], {a});
}
Node Graph::pad(Node a, int axis, int before, int after) {
  const int rank = static_cast<int>(impl_->at(a).shape.count);
  const int ax = impl_->axis(a, axis);
  Shape left(rank, 0), right(rank, 0);
  left[ax] = before;
  right[ax] = after;
  return impl_->put([impl_->g padTensor:impl_->at(a)
                        withPaddingMode:MPSGraphPaddingModeConstant
                            leftPadding:ns_shape(left)
                           rightPadding:ns_shape(right)
                          constantValue:0.0
                                   name:nil],
                     {a});
}
Node Graph::concat(const std::vector<Node>& parts, int axis) {
  NSMutableArray<MPSGraphTensor*>* ts = [NSMutableArray array];
  for (Node p : parts) [ts addObject:impl_->at(p)];
  return impl_->put([impl_->g concatTensors:ts dimension:impl_->axis(parts.at(0), axis) name:nil], parts);
}
Node Graph::one_hot(Node ids, int depth) {
  MPSGraph* g = impl_->g;
  MPSGraphTensor* i = impl_->at(ids);
  // Out-of-range ids: clamped for the lookup, then zeroed by a mask, so the
  // result does not rest on what MPSGraph does with them.
  MPSGraphTensor* lo = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
  MPSGraphTensor* hi = [g constantWithScalar:depth - 1 dataType:MPSDataTypeInt32];
  MPSGraphTensor* clamped = [g clampWithTensor:i minValueTensor:lo maxValueTensor:hi name:nil];
  MPSGraphTensor* hot = [g oneHotWithIndicesTensor:clamped depth:static_cast<NSUInteger>(depth) dataType:MPSDataTypeFloat32 name:nil];
  MPSGraphTensor* ok = [g equalWithPrimaryTensor:i secondaryTensor:clamped name:nil];
  ok = [g castTensor:ok toType:MPSDataTypeFloat32 name:@"in_range"];
  ok = [g expandDimsOfTensor:ok axis:-1 name:nil];
  return impl_->put([g multiplicationWithPrimaryTensor:hot secondaryTensor:ok name:nil], {ids});
}
Node Graph::gather(Node a, Node ids, int axis) {
  return impl_->put([impl_->g gatherWithUpdatesTensor:impl_->at(a)
                                        indicesTensor:impl_->at(ids)
                                                 axis:static_cast<NSUInteger>(impl_->axis(a, axis))
                                      batchDimensions:0
                                                 name:nil],
                     {a, ids});
}
Node Graph::softmax_ce(Node logits, Node target) {
  MPSGraphTensor* l = impl_->at(logits);
  MPSGraphTensor* ce = [impl_->g softMaxCrossEntropyWithSourceTensor:l
                                                       labelsTensor:impl_->at(target)
                                                               axis:-1
                                                      reductionType:MPSGraphLossReductionTypeNone
                                                               name:nil];
  NSMutableArray<NSNumber*>* shape = [l.shape mutableCopy];
  shape[shape.count - 1] = @1;
  return impl_->put([impl_->g reshapeTensor:ce withShape:shape name:nil], {logits, target});
}
Node Graph::softmax(Node a, int axis) {
  return impl_->put([impl_->g softMaxWithTensor:impl_->at(a) axis:impl_->axis(a, axis) name:nil], {a});
}

std::vector<Node> Graph::gradients(Node loss, const std::vector<Node>& wrt) {
  // What the loss depends on.
  std::vector<bool> used(impl_->parents.size(), false);
  std::vector<int> todo{loss.id};
  while (!todo.empty()) {
    const int n = todo.back();
    todo.pop_back();
    if (n < 0 || used[static_cast<std::size_t>(n)]) continue;
    used[static_cast<std::size_t>(n)] = true;
    for (int p : impl_->parents[static_cast<std::size_t>(n)]) todo.push_back(p);
  }
  NSMutableArray<MPSGraphTensor*>* ts = [NSMutableArray array];
  for (Node n : wrt) {
    if (used[static_cast<std::size_t>(n.id)]) [ts addObject:impl_->at(n)];
  }
  NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
      ts.count > 0 ? [impl_->g gradientForPrimaryTensor:impl_->at(loss) withTensors:ts name:nil] : @{};
  std::vector<Node> out;
  out.reserve(wrt.size());
  for (Node n : wrt) {
    MPSGraphTensor* d = used[static_cast<std::size_t>(n.id)] ? grads[impl_->at(n)] : nil;
    out.push_back(d != nil ? impl_->put(d, {loss, n}) : zeros(shape(n)));
  }
  return out;
}

void Graph::run(const std::vector<Bind>& inputs, const std::vector<Bind>& outputs) {
  // The tensor data last made for a buffer, kept while the shape stays.
  const auto tensor_data = [](Buffer::Impl* b, MPSGraphTensor* t) {
    NSArray<NSNumber*>* shape = t.shape;
    if (b->data == nil || ![b->data_shape isEqualToArray:shape]) {
      const MPSDataType type = b->type == Buffer::Type::kInt ? MPSDataTypeInt32 : MPSDataTypeFloat32;
      b->data = [[MPSGraphTensorData alloc] initWithMTLBuffer:b->buffer shape:shape dataType:type];
      b->data_shape = shape;
    }
    return b->data;
  };
  @autoreleasepool {
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [NSMutableDictionary dictionary];
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [NSMutableDictionary dictionary];
    const auto bind = [&](const Bind& b, NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* into) {
      MPSGraphTensor* t = impl_->at(b.node);
      if (b.buffer.empty() || b.buffer.size() != count(t.shape)) {
        throw std::invalid_argument("zf::apple: buffer of " + std::to_string(b.buffer.size()) +
                                    " for a tensor of " + std::to_string(count(t.shape)));
      }
      const bool ints = b.buffer.type() == Buffer::Type::kInt;
      if (ints != (t.dataType == MPSDataTypeInt32)) throw std::invalid_argument("zf::apple: buffer type");
      into[t] = tensor_data(b.buffer.impl_.get(), t);
    };
    for (const Bind& b : inputs) bind(b, feeds);
    for (const Bind& b : outputs) bind(b, results);
    [impl_->g runWithMTLCommandQueue:impl_->queue feeds:feeds targetOperations:nil resultsDictionary:results];
  }
}

}  // namespace apple
}  // namespace zf
