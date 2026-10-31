#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

// bfloat16: the top half of an IEEE float -- same sign and 8-bit exponent, 7
// mantissa bits. Same range as float, about three significant digits.
//
// A storage type. Reading one gives a float (exactly: widening is a shift),
// and arithmetic happens in float. Writing one back is explicit and rounds to
// nearest even:
//
//   float f = x;             // fine, exact
//   bf16 y = bf16(f * 2.0f); // explicit, rounds once
//   bf16 z = x * x;          // does not compile: where would it round?
//
// There are deliberately no arithmetic operators, so every place a value is
// narrowed back to 16 bits is written out. In the models that is where a
// result is stored; sums stay in float (accumulator_t<bf16>).
//
// The layout is one uint16_t, the same bits as the hardware's bfloat16, so
// a buffer of these can be handed to NEON BF16 or AVX512-BF16 kernels as is.

namespace zf {

class bf16 {
 public:
  bf16() = default;

  constexpr explicit bf16(float f) : bits_(round_from(f)) {}

  // From any other arithmetic type, through float. Lets generic code write
  // T{1} and static_cast<T>(0.5) for either element type.
  template <typename A,
            std::enable_if_t<std::is_arithmetic_v<A> &&
                                 !std::is_same_v<A, float>, int> = 0>
  constexpr explicit bf16(A v) : bf16(static_cast<float>(v)) {}

  constexpr operator float() const { return widen(bits_); }

  static constexpr bf16 from_bits(std::uint16_t b) {
    bf16 v;
    v.bits_ = b;
    return v;
  }
  constexpr std::uint16_t bits() const { return bits_; }

  // Float bits -> bf16 bits, round to nearest even. A NaN stays a NaN: the
  // rounding increment could otherwise carry it into infinity. Written as a
  // select, not an early return, so a loop of these vectorizes -- with the
  // branch, narrowing a tensor ran at one element per cycle.
  static constexpr std::uint16_t round_from(float f) {
    const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t lsb = (u >> 16) & 1u;
    const std::uint32_t rounded = (u + 0x7fffu + lsb) >> 16;
    const std::uint32_t quiet_nan = (u >> 16) | 0x0040u;
    const bool nan = (u & 0x7fffffffu) > 0x7f800000u;
    return static_cast<std::uint16_t>(nan ? quiet_nan : rounded);
  }

  static constexpr float widen(std::uint16_t b) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
  }

 private:
  std::uint16_t bits_ = 0;
};

static_assert(sizeof(bf16) == 2, "bf16 must be exactly 16 bits");
static_assert(std::is_trivially_copyable_v<bf16>);

// Element types the trainable layers accept: real numbers, not integers.
template <typename T>
inline constexpr bool is_real_v =
    std::is_floating_point_v<T> || std::is_same_v<T, bf16>;

}  // namespace zf

template <>
class std::numeric_limits<zf::bf16> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr int digits = 8;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int max_exponent = 128;
  static constexpr zf::bf16 min() { return zf::bf16::from_bits(0x0080); }
  static constexpr zf::bf16 lowest() { return zf::bf16::from_bits(0xff7f); }
  static constexpr zf::bf16 max() { return zf::bf16::from_bits(0x7f7f); }
  static constexpr zf::bf16 epsilon() { return zf::bf16::from_bits(0x3c00); }
  static constexpr zf::bf16 infinity() { return zf::bf16::from_bits(0x7f80); }
  static constexpr zf::bf16 quiet_NaN() { return zf::bf16::from_bits(0x7fc0); }
};
