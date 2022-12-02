#include <immintrin.h> 

#ifdef _MSC_VER
#include <intrin.h>
#    ifndef AXPOPCNT32(x)
#        define AXPOPCNT32(x) __popcnt(x)
#    endif
#    ifndef AXPOPCNT64(x)
#        define AXPOPCNT64(x) __popcnt64(x)
#    endif
#elif defined(__GNUC__) && !defined(__MINGW32__)
#    ifndef AXPOPCNT32(x)
#        define AXPOPCNT32(x) __builtin_popcount(x)
#    endif
#    ifndef AXPOPCNT64(x)
#        define AXPOPCNT64(x) __builtin_popcountl(x)
#    endif
#else
#   ifndef AXPOPCNT32(x)
#       define AXPOPCNT32(x) PopCount(x)
#   endif
#   ifndef AXPOPCNT64(x)
#       define AXPOPCNT64(x) PopCount(x)
#   endif
#endif

#ifndef FINLINE
#   ifdef _MSC_VER
#	define FINLINE __forceinline
#   elif __CLANG__
#       define FINLINE [[clang::always_inline]] 
#   elif __GNUC__
#       define FINLINE  __attribute__((always_inline))
#   endif
#endif

#ifndef VECTORCALL
#   ifdef _MSC_VER
#	define VECTORCALL __vectorcall
#   elif __CLANG__
#       define VECTORCALL [[clang::vectorcall]] 
#   elif __GNUC__
#       define VECTORCALL  
#   endif
#endif

using ulong = unsigned long;

inline constexpr ulong PopCount(ulong y) // standard popcount; from wikipedia
{
	y -= ((y >> 1) & 0x5555555555555555ull);
	y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);
	return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}

inline constexpr int PopCount(uint i)
{
	i = i - ((i >> 1) & 0x55555555);        // add pairs of bits
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);  // quads
	i = (i + (i >> 4)) & 0x0F0F0F0F;        // groups of 8
	return (i * 0x01010101) >> 24;          // horizontal sum of bytes
}
// from Faster Population Counts Using AVX2 Instructions resource paper
FINLINE long VECTORCALL popcount256_epi64(__m256i v)
{
	__m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
			2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
			1, 2, 2, 3, 2, 3, 3, 4);
	__m256i low_mask = _mm256_set1_epi8(0x0f);
	__m256i lo =  _mm256_and_si256(v, low_mask);
	__m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
	__m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
	__m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
	__m256i total = _mm256_add_epi8(popcnt1, popcnt2);
	
	v = _mm256_sad_epu8(total, _mm256_setzero_si256());
 
	long sum = _mm256_cvtsi256_si32(v);
	sum += _mm256_cvtsi256_si32(_mm256_permute4x64_epi64(v, _MM_SHUFFLE(1, 1, 1, 1)));
	sum += _mm256_cvtsi256_si32(_mm256_permute4x64_epi64(v, _MM_SHUFFLE(2, 2, 2, 2)));
	sum += _mm256_cvtsi256_si32(_mm256_permute4x64_epi64(v, _MM_SHUFFLE(3, 3, 3, 3)));
	return sum;
}
template<int numBits> struct Bitset
{
	static constexpr int size = (numBits / 64) + 1;

	ulong bits[size] = { 0 };
    // todo constructor from string

	bool Get(int idx) const { return !!(bits[idx / 64] & (1ul << idx & 63)); }

	void Set(int idx) { bits[idx / 64] |= 1ul << (idx & 63); }

	// sets bit to 0
	void Reset(int idx) { bits[idx / 64] &= ~(1ul << (idx & 63)); }

	void Clear() {
		for (int i = 0; i < size; ++i) bits[i] = 0ul;
	}

	void Flip() {
		for (int i = 0; i < size - 1; ++i)
			bits[i] = ~bits[i];
		int lastBits = (1ul << (numBits & 63)) - 1;
		bits[size - 1] ^= lastBits;	
	}

	bool All() const {
		for (int i = 0; i < size - 1; ++i)
			if (bits[i] != ~0ul) return false;
		int lastBits = (1ul << (numBits & 63)) - 1;
		return (bits[size - 1] & lastBits) == lastBits;
	}

	bool Any() const {
		for (int i = 0; i < size-1; ++i)
			if (bits[i] > 0ul) return true;
		int lastBits = (1ul << (numBits & 63)) - 1;
		return (bits[size-1] & lastBits) > 0ul;
	}

	int Count() const {
		int sum = 0;
		for (int i = 0; i < size-1; ++i)
		    sum += AXPOPCNT64(bits[i]);
        
		int lastBits = numBits & 63;
		for (int i = 0; i < lastBits; ++i)
			sum += ((1ul << i) & bits[size-1]) > 0;
		return sum;
	}

	// std::string toString()
	// {
	//  std::string str; str.resize(numBits);
	// 	for (int i = 0, currBit = 0; i < size; ++i)
	// 		for (int j = 0; j < 64 && currBit < numBits; ++j, ++currBit)
	// 			str[currBit] = (bits[i] & (1ul << j) ? '1' : '0');
	// 	return str;
	// }
};
template<typename T>
inline constexpr void FillN(T* ptr, int len, T val) {
	for (int i = 0; i < len; ++i)
		ptr[i] = val;
}

struct Bitset128
{
	unsigned long bits[2] = {};

	Bitset128() { bits[0] = 0ul; bits[1] = 0ul; }
	Bitset128(unsigned long repeat) { bits[0] = repeat; bits[1] = repeat;  }
	Bitset128(ulong a, ulong b) { bits[0] = a; bits[1] = b; }
	bool Get(int idx) const { return !!(bits[idx > 63] & (1ul << (idx & 63ul))); }
	void Set(int idx) { bits[idx > 63] |= 1ul << (idx & 63); }

	Bitset128 operator &  (Bitset128& other) const { return { bits[0] & other.bits[0], bits[1] & other.bits[1] }; }
	Bitset128 operator |  (Bitset128& other) const { return { bits[0] | other.bits[0], bits[1] | other.bits[1] }; }
	Bitset128 operator ^  (Bitset128& other) const { return { bits[0] ^ other.bits[0], bits[1] ^ other.bits[1] }; }

	Bitset128 operator &=  (Bitset128& other) { bits[0] &= other.bits[0], bits[1] &= other.bits[1]; return *this; }
	Bitset128 operator |=  (Bitset128& other) { bits[0] |= other.bits[0], bits[1] |= other.bits[1]; return *this; }
	Bitset128 operator ^=  (Bitset128& other) { bits[0] ^= other.bits[0], bits[1] ^= other.bits[1]; return *this; }

	// sets bit to 0
	void Reset(int idx) { bits[idx > 63] &= ~(1ul << (idx & 63)); }

	void Clear() { bits[0] = 0ul, bits[1] = 0ul;  }
	void Flip()  { bits[0] = ~bits[0], bits[1] = ~bits[1]; }
	
	bool All()  const { return bits[0] == ~0ul && bits[1] == ~0ul; }
	bool Any()  const { return bits[0] + bits[1] > 0; }
	int Count() const { return AXPOPCNT64(bits[0]) + AXPOPCNT64(bits[1]); }
};

struct Bitset256
{
	union {
		unsigned long bits[4] = {};
		__m256i sse;
	};
	Bitset256() { Clear(); }
	Bitset256(ulong r) { bits[0] = r; bits[1] = r; bits[2] = r; bits[3] = r; }
	Bitset256(ulong a, ulong b, ulong c, ulong d) { bits[0] = a; bits[1] = b; bits[2] = c; bits[3] = d; }
	Bitset256(__m256i v) : sse(v) {}
	
	bool Get(int idx) const { return !!(bits[idx > 63] & (1ul << (idx & 63ul))); }
	void Set(int idx) { bits[idx > 63] |= 1ul << (idx & 63); }
	void Clear() { sse = _mm256_set_epi64x(0ul, 0ul, 0ul, 0ul);  }
	void Flip() { sse = _mm256_xor_si256(sse, _mm256_set1_epi32(0xffffffff)); }
	Bitset256 operator~ () { return { _mm256_xor_si256(sse, _mm256_set1_epi32(0xffffffff)) }; }

	Bitset256 operator &  (Bitset256& other) const { return _mm256_and_si256(sse, other.sse); }
	Bitset256 operator |  (Bitset256& other) const { return _mm256_or_si256 (sse, other.sse); }
	Bitset256 operator ^  (Bitset256& other) const { return _mm256_xor_si256(sse, other.sse); }

	Bitset256& operator &=  (Bitset256& other) { sse = _mm256_and_si256(sse, other.sse); return *this; }
	Bitset256& operator |=  (Bitset256& other) { sse = _mm256_or_si256(sse, other.sse);  return *this; }
	Bitset256& operator ^=  (Bitset256& other) { sse = _mm256_xor_si256(sse, other.sse); return *this; }

	bool All() const {
		return _mm256_movemask_epi8(_mm256_cmpeq_epi64(sse,
			_mm256_set_epi64x(~0ul, ~0ul, ~0ul, ~0ul))) == ~0ul;
	}
	bool Any() const {
		return _mm256_movemask_epi8(_mm256_cmpeq_epi64(sse,
			_mm256_set_epi64x(~0ul, ~0ul, ~0ul, ~0ul))) > 0;
	}
	
	int Count() const { return popcount256_epi64(sse); }
};

#define _AND(a, b) _mm256_and_si256(a, b)
#define _ROR(a, b) _mm256_or_si256 (a, b)
#define _XOR(a, b) _mm256_xor_si256(a, b)

struct Bitset512
{
	union {
		unsigned long bits[8] = {};
		__m256i sse[2];
	};

	Bitset512() { Clear(); }
	Bitset512(unsigned long repeat) { FillN(bits, 8, repeat); }
	Bitset512(__m256i a, __m256i b) { sse[0] = a; sse[1] = b; }
	Bitset512 operator~ () { return Bitset512(_mm256_xor_si256(sse[0], _mm256_set1_epi32(0xffffffff)), _mm256_xor_si256(sse[1], _mm256_set1_epi32(0xffffffff))); }

	Bitset512 operator &  (const Bitset512 o) const { return { _mm256_and_si256(sse[0], o.sse[0]), _mm256_and_si256(sse[1], o.sse[1]) }; }
	Bitset512 operator |  (const Bitset512 o) const { return { _mm256_or_si256 (sse[0], o.sse[0]), _mm256_or_si256 (sse[1], o.sse[1]) }; }
	Bitset512 operator ^  (const Bitset512 o) const { return { _mm256_xor_si256(sse[0], o.sse[0]), _mm256_xor_si256(sse[1], o.sse[1]) }; }

	Bitset512 operator &=  (const Bitset512 o) { sse[0] = _mm256_and_si256(sse[0], o.sse[0]); sse[1] = _mm256_and_si256(sse[1], o.sse[1]); return *this; }
	Bitset512 operator |=  (const Bitset512 o) { sse[0] = _mm256_or_si256 (sse[0], o.sse[0]); sse[1] = _mm256_or_si256 (sse[1], o.sse[1]); return *this; }
	Bitset512 operator ^=  (const Bitset512 o) { sse[0] = _mm256_xor_si256(sse[0], o.sse[0]); sse[1] = _mm256_xor_si256(sse[1], o.sse[1]); return *this; }

	void And(Bitset512& o) { sse[0] = _mm256_and_si256(sse[0], o.sse[0]); sse[1] = _mm256_and_si256(sse[1], o.sse[1]); }
	void Or (Bitset512& o) { sse[0] = _mm256_or_si256 (sse[0], o.sse[0]); sse[1] = _mm256_or_si256 (sse[1], o.sse[1]); }
	void Xor(Bitset512& o) { sse[0] = _mm256_xor_si256(sse[0], o.sse[0]); sse[1] = _mm256_xor_si256(sse[1], o.sse[1]); }

	bool Get(int idx) const { return !!(bits[idx / 64] & (1ul << (idx & 63ul))); }
	void Set(int idx) { bits[idx / 64] |= 1ul << (idx & 63); }
	void Reset(int idx) { bits[idx / 64] &= ~(1ul << (idx & 63)); }

	void Clear() { sse[0] = sse[1] = _mm256_set_epi64x(0ul, 0ul, 0ul, 0ul); }
	void Flip()  {
		sse[0] = _mm256_xor_si256(sse[0], _mm256_set1_epi32(0xffffffff)); 
		sse[1] = _mm256_xor_si256(sse[1], _mm256_set1_epi32(0xffffffff));
	}

	bool All() const {
		const __m256i full = _mm256_set_epi64x(~0ul, ~0ul, ~0ul, ~0ul);
		return _mm256_movemask_epi8(_mm256_cmpeq_epi64(sse[0], full)) == ~0ul &&
			   _mm256_movemask_epi8(_mm256_cmpeq_epi64(sse[1], full)) == ~0ul;
	}
	bool Any() const {
		const __m256i full = _mm256_set_epi64x(~0ul, ~0ul, ~0ul, ~0ul);
		return _mm256_movemask_epi8(_mm256_cmpeq_epi64(sse[0], full)) > 0 &&
			   _mm256_movemask_epi8(_mm256_cmpeq_epi64(sse[1], full));
	}

	int Count() const {
		return popcount256_epi64(sse[0]) + popcount256_epi64(sse[1]);
	}
};

struct Bitset1024
{
	union {
		unsigned long bits[16] = {};
		struct { Bitset512 b1, b2; };
		struct { __m256i v[4]; };
	};
	
	Bitset1024() { Clear(); }
	Bitset1024(unsigned long repeat) { FillN(bits, 16, repeat); }
	Bitset1024(Bitset512 a, Bitset512 b) { b1 = (Bitset512&&)a; b2 = (Bitset512&&)b; }

	bool Get(int idx)  const { return !!(bits[idx / 64] & (1ul << (idx & 63ul))); }
	void Set(int idx)   { bits[idx / 64] |= 1ul << (idx & 63); }
	void Reset(int idx) { bits[idx / 64] &= ~(1ul << (idx & 63)); }
	Bitset1024 operator~  () { return {~b1, ~b2}; }

	Bitset1024 operator &  (const Bitset1024& o) const { Bitset1024 r; r.v[0] = _AND(v[0], o.v[0]); r.v[1] = _AND(v[1], o.v[1]); r.v[2] = _AND(v[2], o.v[2]); r.v[3] = _AND(v[3], o.v[3]); return r; }
	Bitset1024 operator |  (const Bitset1024& o) const { Bitset1024 r; r.v[0] = _ROR(v[0], o.v[0]); r.v[1] = _ROR(v[1], o.v[1]); r.v[2] = _ROR(v[2], o.v[2]); r.v[3] = _ROR(v[3], o.v[3]); return r; }
	Bitset1024 operator ^  (const Bitset1024& o) const { Bitset1024 r; r.v[0] = _XOR(v[0], o.v[0]); r.v[1] = _XOR(v[1], o.v[1]); r.v[2] = _XOR(v[2], o.v[2]); r.v[3] = _XOR(v[3], o.v[3]); return r; }

	Bitset1024& operator &=  (const Bitset1024& o) { v[0] = _AND(v[0], o.v[0]); v[1] = _AND(v[1], o.v[1]); v[2] = _AND(v[2], o.v[2]); v[3] = _AND(v[3], o.v[3]); return *this; }
	Bitset1024& operator |=  (const Bitset1024& o) { v[0] = _ROR(v[0], o.v[0]); v[1] = _ROR(v[1], o.v[1]); v[2] = _ROR(v[2], o.v[2]); v[3] = _ROR(v[3], o.v[3]); return *this; }
	Bitset1024& operator ^=  (const Bitset1024& o) { v[0] = _XOR(v[0], o.v[0]); v[1] = _XOR(v[1], o.v[1]); v[2] = _XOR(v[2], o.v[2]); v[3] = _XOR(v[3], o.v[3]); return *this; }

	void And(Bitset1024& o) { v[0] = _AND(v[0], o.v[0]); v[1] = _AND(v[1], o.v[1]); v[2] = _AND(v[2], o.v[2]); v[3] = _AND(v[3], o.v[3]); }
	void Or(Bitset1024&  o) { v[0] = _ROR(v[0], o.v[0]); v[1] = _ROR(v[1], o.v[1]); v[2] = _ROR(v[2], o.v[2]); v[3] = _ROR(v[3], o.v[3]); }
	void Xor(Bitset1024& o) { v[0] = _XOR(v[0], o.v[0]); v[1] = _XOR(v[1], o.v[1]); v[2] = _XOR(v[2], o.v[2]); v[3] = _XOR(v[3], o.v[3]); }

	void Clear() { v[0] = v[1] = v[2] = v[3] = _mm256_setzero_si256();  }
	void Flip()  { b1.Flip(), b2.Flip(); }
	bool All() const {
		const __m256i full = _mm256_set_epi64x(~0ul, ~0ul, ~0ul, ~0ul);
		return _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[0], full)) == ~0ul && _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[1], full)) == ~0ul && _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[2], full)) == ~0ul && _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[3], full)) == ~0ul;
	}
	bool Any() const {
		const __m256i full = _mm256_set_epi64x(~0ul, ~0ul, ~0ul, ~0ul);
		return _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[0], full)) > 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[1], full)) > 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[2], full)) > 0 && _mm256_movemask_epi8(_mm256_cmpeq_epi64(v[3], full)) > 0;
	}
	int Count() const { return popcount256_epi64(v[0]) + popcount256_epi64(v[1]) + popcount256_epi64(v[2]) + popcount256_epi64(v[3]); }
};

struct Bitset2048
{
	union {
		unsigned long bits[32] = {};
		struct { Bitset1024 b1, b2; };
		struct { __m256i v[8]; };
	};
	
	Bitset2048() { Clear(); }
	Bitset2048(unsigned long repeat) { FillN(bits, 32, repeat); }
	Bitset2048(Bitset1024 a, Bitset1024 b) { b1 = a; b2 = b; }

	Bitset2048 operator~  () { return {~b1, ~b2}; }

	Bitset2048 operator &  (const Bitset2048& o) const { Bitset2048 r; r.v[0] = _AND(v[0], o.v[0]); r.v[1] = _AND(v[1], o.v[1]); r.v[2] = _AND(v[2], o.v[2]); r.v[3] = _AND(v[3], o.v[3]); r.v[4] = _AND(v[4], o.v[4]); r.v[5] = _AND(v[5], o.v[5]); r.v[6] = _AND(v[6], o.v[6]); r.v[7] = _AND(v[7], o.v[7]); return r; }
	Bitset2048 operator |  (const Bitset2048& o) const { Bitset2048 r; r.v[0] = _ROR(v[0], o.v[0]); r.v[1] = _ROR(v[1], o.v[1]); r.v[2] = _ROR(v[2], o.v[2]); r.v[3] = _ROR(v[3], o.v[3]); r.v[4] = _ROR(v[4], o.v[4]); r.v[5] = _ROR(v[5], o.v[5]); r.v[6] = _ROR(v[6], o.v[6]); r.v[7] = _ROR(v[7], o.v[7]); return r; }
	Bitset2048 operator ^  (const Bitset2048& o) const { Bitset2048 r; r.v[0] = _XOR(v[0], o.v[0]); r.v[1] = _XOR(v[1], o.v[1]); r.v[2] = _XOR(v[2], o.v[2]); r.v[3] = _XOR(v[3], o.v[3]); r.v[4] = _XOR(v[4], o.v[4]); r.v[5] = _XOR(v[5], o.v[5]); r.v[6] = _XOR(v[6], o.v[6]); r.v[7] = _XOR(v[7], o.v[7]); return r; }

	Bitset2048& operator &=  (const Bitset2048& o) { v[0] = _AND(v[0], o.v[0]); v[1] = _AND(v[1], o.v[1]); v[2] = _AND(v[2], o.v[2]); v[3] = _AND(v[3], o.v[3]); v[4] = _AND(v[4], o.v[4]); v[5] = _AND(v[5], o.v[5]); v[6] = _AND(v[6], o.v[6]); v[7] = _AND(v[7], o.v[7]); return *this; }
	Bitset2048& operator |=  (const Bitset2048& o) { v[0] = _ROR(v[0], o.v[0]); v[1] = _ROR(v[1], o.v[1]); v[2] = _ROR(v[2], o.v[2]); v[3] = _ROR(v[3], o.v[3]); v[4] = _ROR(v[4], o.v[4]); v[5] = _ROR(v[5], o.v[5]); v[6] = _ROR(v[6], o.v[6]); v[7] = _ROR(v[7], o.v[7]); return *this; }
	Bitset2048& operator ^=  (const Bitset2048& o) { v[0] = _XOR(v[0], o.v[0]); v[1] = _XOR(v[1], o.v[1]); v[2] = _XOR(v[2], o.v[2]); v[3] = _XOR(v[3], o.v[3]); v[4] = _XOR(v[4], o.v[4]); v[5] = _XOR(v[5], o.v[5]); v[6] = _XOR(v[6], o.v[6]); v[7] = _XOR(v[7], o.v[7]); return *this; }

	void And(Bitset2048& o) { v[0] = _AND(v[0], o.v[0]); v[1] = _AND(v[1], o.v[1]); v[2] = _AND(v[2], o.v[2]); v[3] = _AND(v[3], o.v[3]); v[4] = _AND(v[4], o.v[4]); v[5] = _AND(v[5], o.v[5]); v[6] = _AND(v[6], o.v[6]); v[7] = _AND(v[7], o.v[7]); }
	void Or (Bitset2048& o) { v[0] = _ROR(v[0], o.v[0]); v[1] = _ROR(v[1], o.v[1]); v[2] = _ROR(v[2], o.v[2]); v[3] = _ROR(v[3], o.v[3]); v[4] = _ROR(v[4], o.v[4]); v[5] = _ROR(v[5], o.v[5]); v[6] = _ROR(v[6], o.v[6]); v[7] = _ROR(v[7], o.v[7]); }
	void Xor(Bitset2048& o) { v[0] = _XOR(v[0], o.v[0]); v[1] = _XOR(v[1], o.v[1]); v[2] = _XOR(v[2], o.v[2]); v[3] = _XOR(v[3], o.v[3]); v[4] = _XOR(v[4], o.v[4]); v[5] = _XOR(v[5], o.v[5]); v[6] = _XOR(v[6], o.v[6]); v[7] = _XOR(v[7], o.v[7]); }

	bool Get(int idx) const { return !!(bits[idx / 64] & (1ul << (idx & 63ul))); }
	void Set(int idx) { bits[idx / 64] |= 1ul << (idx & 63); }
	void Reset(int idx) { bits[idx / 64] &= ~(1ul << (idx & 63)); }

	void Clear() { b1.Clear(), b2.Clear(); }
	void Flip()  { b1.Flip(), b2.Flip(); }
	bool All()  const { return b1.All() && b2.All(); }
	bool Any()  const { return b1.Any() && b2.Any(); }
	int Count() const { return b1.Count() + b2.Count(); }
};

#undef _AND
#undef _ROR
#undef _XOR
