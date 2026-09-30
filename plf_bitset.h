// Copyright (c) 2026, Matthew Bentley (mattreecebentley@gmail.com) www.plflib.org

// Computing For Good License v1.02 (https://plflib.org/computing_for_good_license.htm):
// This code is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this code.
//
// Permission is granted to use this code by anyone and for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:
//
// 1. 	The origin of this code must not be misrepresented; you must not claim that you wrote the original code. If you use this code in software, an acknowledgement in the product documentation would be appreciated but is not required.
// 2. 	Altered code versions must be plainly marked as such, and must not be misrepresented as being the original code.
// 3. 	This notice may not be removed or altered from any code distribution, including altered code versions.
// 4. 	This code and altered code versions may not be used by groups, companies, individuals or in software whose primary or partial purpose is to:
// 	 a.	 Promote addiction.
// 	 b.	 Cause harm to, or violate the rights of, other sentient beings.
// 	 c.	 Distribute, obtain or utilize software, media or other materials without the consent of the owners.
// 	 d.	 Deliberately spread misinformation or encourage dishonesty.
// 	 e.	 Pursue personal profit at the cost of broad-scale environmental harm.




#ifndef PLF_BITSET_H
#define PLF_BITSET_H


#ifndef PLF_COMPILER_DEFINES
	#define PLF_BITSET_DEFINES // ie. No encapsulating unit/class has previously defined the compiler feature macros in plf_tools.h below, so allow this header to undefine them at it's end.
#endif

#define PLF_INCLUDE_BIT_TOOLS
#define PLF_INCLUDE_TOOLS
#include "plf_tools.h"


#define PLF_TYPE_BITWIDTH (sizeof(storage_type) * 8)
#define PLF_ARRAY_CAPACITY ((total_size + PLF_TYPE_BITWIDTH - 1) / PLF_TYPE_BITWIDTH) // ie. round up to nearest unit of storage
#define PLF_ARRAY_CAPACITY_BITS (PLF_ARRAY_CAPACITY * PLF_TYPE_BITWIDTH)
#define PLF_ARRAY_CAPACITY_BYTES (PLF_ARRAY_CAPACITY * sizeof(storage_type))

#if defined(__clang__) // Stops clang vectorizing a loop across its iterations, which it otherwise sometimes does for the 8-at-a-time search loops below using gather instructions, which are much slower than the unvectorized loop
	#define PLF_NO_VECTORIZE _Pragma("clang loop vectorize(disable)")
#else
	#define PLF_NO_VECTORIZE
#endif


#include <cmath> // log10
#include <cassert>
#include <string>	// std::basic_string
#include <stdexcept> // std::out_of_range
#include <limits>  // std::numeric_limits
#include <ostream>
#include <cstring>	// memset, size_t
#include <algorithm> // std::copy, std::equal

#ifdef PLF_CPP20_SUPPORT
	#include <bit>  // std::pop_count, std::countr_one, std::countr_zero
#endif



namespace plf
{


template<std::size_t total_size, typename storage_type = std::size_t, bool hardened = false>
class bitset
{
private:
	typedef std::size_t size_type;
	storage_type buffer[PLF_ARRAY_CAPACITY];


	// These two function calls should be optimized out by the compiler (under C++20) if total_size is a multiple of storage_type bitwidth, but if the "if" statement can't be constexpr due to lack of C++20 support, avoid the CPU penalty of the branch instruction and just perform the operation anyway. The idea is that there may be some remainder in the final storage_type which is unused in the bitset. By default we keep this at 0 for all bits, however some operations require them to be 1 in order to perform optimally. For those operations we set the remainder (overflow) to 1, then back to 0 at the end of the function:

	PLF_CONSTFUNC void set_overflow_to_one() PLF_NOEXCEPT
	{ // If total_size < array bit capacity, set all bits > size to 1
		#ifdef PLF_CPP20_SUPPORT
			if constexpr (total_size % PLF_TYPE_BITWIDTH != 0)
			{
				buffer[PLF_ARRAY_CAPACITY - 1] |= std::numeric_limits<storage_type>::max() << (PLF_TYPE_BITWIDTH - (PLF_ARRAY_CAPACITY_BITS - total_size));
			}
		#else // Can't remove the code if total_size % PLF_TYPE_BITWIDTH == 0, so avoid the branch instead:
			buffer[PLF_ARRAY_CAPACITY - 1] |= static_cast<storage_type>(~(std::numeric_limits<storage_type>::max() >> (PLF_ARRAY_CAPACITY_BITS - total_size))); // very slightly slower op based on benchmarking
		#endif
	}



	PLF_CONSTFUNC void set_overflow_to_zero() PLF_NOEXCEPT
	{ // If total_size < array bit capacity, set all bits > size to 0
		#ifdef PLF_CPP20_SUPPORT
			if constexpr (total_size % PLF_TYPE_BITWIDTH != 0)
		#endif
		{
			buffer[PLF_ARRAY_CAPACITY - 1] &= std::numeric_limits<storage_type>::max() >> (PLF_ARRAY_CAPACITY_BITS - total_size);
		}
	}



	PLF_CONSTFUNC storage_type last_word_with_overflow_set() const PLF_NOEXCEPT
	{ // Returns the final storage_type with all bits > size set to 1, without modifying the buffer - allows const functions to use the same optimisation as set_overflow_to_one()
		return static_cast<storage_type>(buffer[PLF_ARRAY_CAPACITY - 1] | ~(std::numeric_limits<storage_type>::max() >> (PLF_ARRAY_CAPACITY_BITS - total_size)));
	}



	// These find the first (or last) storage_type which isn't equal to skip_value. Combining storage_type's with XOR/OR before comparing is significantly faster than comparing each individually, so they check 8 storage_type's at a time, and 64 bytes at a time for storage types narrower than 8 bytes. A nearby result is checked for before going 64 bytes at a time, so it doesn't cost a full 64-byte check:

	template <storage_type skip_value>
	static PLF_CONSTFUNC bool group_differs(const storage_type * const group) PLF_NOEXCEPT
	{ // ie. whether any of the 8 storage_type's starting at group differ from skip_value
		return ((group[0] ^ skip_value) | (group[1] ^ skip_value) | (group[2] ^ skip_value) | (group[3] ^ skip_value) | (group[4] ^ skip_value) | (group[5] ^ skip_value) | (group[6] ^ skip_value) | (group[7] ^ skip_value)) != 0;
	}



	template <storage_type skip_value>
	static PLF_CONSTFUNC const storage_type * locate_forwards(const storage_type *current) PLF_NOEXCEPT
	{ // Only called when a storage_type differing from skip_value is known to be at or after current
		while (*current == skip_value) ++current;
		return current;
	}



	template <storage_type skip_value>
	static PLF_CONSTFUNC const storage_type * locate_backwards(const storage_type *current) PLF_NOEXCEPT
	{ // Only called when a storage_type differing from skip_value is known to be at or before current
		while (*current == skip_value) --current;
		return current;
	}



	template <storage_type skip_value>
	static PLF_CONSTFUNC const storage_type * find_word_forwards(const storage_type *current, const storage_type * const end) PLF_NOEXCEPT
	{ // Returns the first storage_type in [current, end) which differs from skip_value, or end if there isn't one
		const size_type block_size = 64 / sizeof(storage_type);

		if (end - current < 8)
		{
			while (current != end && *current == skip_value) ++current;
			return current;
		}

		if PLF_CONSTEXPR (block_size > 8)
		{
			if (group_differs<skip_value>(current)) return locate_forwards<skip_value>(current);
			current += 8;

			for (; static_cast<size_type>(end - current) >= block_size; current += block_size)
			{
				storage_type combined = 0;
				for (size_type index = 0; index != block_size; ++index) combined |= current[index] ^ skip_value;
				if (combined != 0) break; // The group loop below finds the storage_type within this block
			}
		}

		PLF_NO_VECTORIZE
		for (; end - current >= 8; current += 8)
		{
			if (group_differs<skip_value>(current)) break; // The storage_type is located after the loop
		}

		if (end - current < 8)
		{
			if (current == end) return end;

			// Check the remaining storage_type's as one group ending at end. This overlaps storage_type's which are already known to equal skip_value, but is faster than checking the remainder individually:
			current = end - 8;
			if (!group_differs<skip_value>(current)) return end;
		}

		return locate_forwards<skip_value>(current);
	}



	template <storage_type skip_value>
	static PLF_CONSTFUNC bool all_words_equal(const storage_type *current, const storage_type * const end) PLF_NOEXCEPT
	{ // ie. find_word_forwards(current, end) == end, but without the cost of locating a differing storage_type once one is known to exist
		const size_type block_size = 64 / sizeof(storage_type);

		if (end - current < 8)
		{
			for (; current != end; ++current)
			{
				if (*current != skip_value) return false;
			}

			return true;
		}

		if PLF_CONSTEXPR (block_size > 8)
		{
			for (; static_cast<size_type>(end - current) >= block_size; current += block_size)
			{
				storage_type combined = 0;
				for (size_type index = 0; index != block_size; ++index) combined |= current[index] ^ skip_value;
				if (combined != 0) return false;
			}
		}

		PLF_NO_VECTORIZE
		for (; end - current >= 8; current += 8)
		{
			if (group_differs<skip_value>(current)) return false;
		}

		return current == end || !group_differs<skip_value>(end - 8); // See find_word_forwards for the overlapping final group
	}



	template <storage_type skip_value>
	static PLF_CONSTFUNC const storage_type * find_word_backwards(const storage_type * const begin, const storage_type *current) PLF_NOEXCEPT
	{ // Returns the last storage_type in [begin, current) which differs from skip_value, or NULL if there isn't one
		const size_type block_size = 64 / sizeof(storage_type);

		if (current - begin < 8)
		{
			while (current != begin)
			{
				if (*--current != skip_value) return current;
			}

			return NULL;
		}

		if PLF_CONSTEXPR (block_size > 8)
		{
			if (group_differs<skip_value>(current - 8)) return locate_backwards<skip_value>(current - 1);
			current -= 8;

			for (; static_cast<size_type>(current - begin) >= block_size; current -= block_size)
			{
				storage_type combined = 0;
				const storage_type * const block = current - block_size;
				for (size_type index = 0; index != block_size; ++index) combined |= block[index] ^ skip_value;
				if (combined != 0) break;
			}
		}

		PLF_NO_VECTORIZE
		for (; current - begin >= 8; current -= 8)
		{
			if (group_differs<skip_value>(current - 8)) break;
		}

		if (current - begin < 8)
		{
			if (current == begin) return NULL;

			// As per find_word_forwards, check the remainder as one group starting at begin:
			current = begin + 8;
			if (!group_differs<skip_value>(begin)) return NULL;
		}

		return locate_backwards<skip_value>(current - 1);
	}



	PLF_CONSTFUNC void check_index_is_within_size(const size_type index) const
	{
		if PLF_CONSTEXPR (hardened)
		{
			if (index >= total_size)
			{
				#ifdef PLF_EXCEPTIONS_SUPPORT
					throw std::out_of_range("Index larger than size of bitset");
				#else
					std::terminate();
				#endif
			}
		}
	}


public:

	PLF_CONSTFUNC bitset() PLF_NOEXCEPT
	{
		reset();
	}



	PLF_CONSTFUNC bitset(const bitset &source) PLF_NOEXCEPT
	{
		std::copy(source.buffer, source.buffer + PLF_ARRAY_CAPACITY, buffer); // copy_n not C++03-compatible
	}



	PLF_CONSTFUNC bool operator [] (const size_type index) const
	{
		if PLF_CONSTEXPR (hardened) check_index_is_within_size(index);
		return static_cast<bool>((buffer[index / PLF_TYPE_BITWIDTH] >> (index % PLF_TYPE_BITWIDTH)) & storage_type(1));
	}



	PLF_CONSTFUNC bool test(const size_type index) const
	{
		if PLF_CONSTEXPR (!hardened) check_index_is_within_size(index); // If hardened, will be checked in []
		return operator [](index);
	}



	PLF_CONSTFUNC void set() PLF_NOEXCEPT
	{
		#ifdef PLF_CONSTEVAL_SUPPORT
			if consteval
			{
				std::fill_n(buffer, PLF_ARRAY_CAPACITY, std::numeric_limits<storage_type>::max()); // fill_n is very slow compared to memset under gcc, particularly in debug mode, but memset isn't constexpr
			}
			else
		#endif
		{
			std::memset(plf::void_cast(buffer), std::numeric_limits<unsigned char>::max(), PLF_ARRAY_CAPACITY_BYTES);
		}

		set_overflow_to_zero();
	}



	PLF_CONSTFUNC void set(const size_type index)
	{
		if PLF_CONSTEXPR (hardened) check_index_is_within_size(index);
		buffer[index / PLF_TYPE_BITWIDTH] |= storage_type(1) << (index % PLF_TYPE_BITWIDTH);
	}



	PLF_CONSTFUNC void set(const size_type index, const bool value)
	{
		if PLF_CONSTEXPR (hardened) check_index_is_within_size(index);

 		const size_type word_index = index / PLF_TYPE_BITWIDTH, shift = index % PLF_TYPE_BITWIDTH;
		buffer[word_index] = (buffer[word_index] & ~(storage_type(1) << shift)) | (static_cast<storage_type>(value) << shift);
	}



	PLF_CONSTFUNC void set_range(const size_type begin, const size_type end)
	{
		if PLF_CONSTEXPR (hardened)
		{
			check_index_is_within_size(begin);
			check_index_is_within_size(end);
		}

		if (begin == end)
		#ifdef PLF_CPP20_SUPPORT
			[[unlikely]]
		#endif
		{
			return;
		}

		const size_type begin_type_index = begin / PLF_TYPE_BITWIDTH, end_type_index = (end - 1) / PLF_TYPE_BITWIDTH, begin_subindex = begin % PLF_TYPE_BITWIDTH, distance_to_end_storage = PLF_TYPE_BITWIDTH - (end % PLF_TYPE_BITWIDTH);

		if (begin_type_index != end_type_index) // ie. if first and last bit to be set are not in the same storage_type unit
		{
			// Write first storage_type:
			buffer[begin_type_index] |= std::numeric_limits<storage_type>::max() << begin_subindex;

			// Fill all intermediate storage_type's (if any):
			#ifdef PLF_CONSTEVAL_SUPPORT
				if consteval
				{
					std::fill_n(buffer + begin_type_index + 1, (end_type_index - 1) - begin_type_index, std::numeric_limits<storage_type>::max());
				}
				else
			#endif
			{
				std::memset(plf::void_cast(buffer + begin_type_index + 1), std::numeric_limits<unsigned char>::max(), ((end_type_index - 1) - begin_type_index) * sizeof(storage_type));
			}

			// Write last storage_type:
			buffer[end_type_index] |= std::numeric_limits<storage_type>::max() >> distance_to_end_storage;
		}
		else
		{
			buffer[begin_type_index] |= (std::numeric_limits<storage_type>::max() << begin_subindex) & (std::numeric_limits<storage_type>::max() >> distance_to_end_storage);
		}
	}



	PLF_CONSTFUNC void set_range(const size_type begin, const size_type end, const bool value)
	{
		if (value)
		{
			set_range(begin, end);
		}
		else
		{
			reset_range(begin, end);
		}
	}



	PLF_CONSTFUNC void reset() PLF_NOEXCEPT
	{
		#ifdef PLF_CONSTEVAL_SUPPORT
			if consteval
			{
				std::fill_n(buffer, PLF_ARRAY_CAPACITY, 0);
			}
			else
		#endif
		{
			std::memset(plf::void_cast(buffer), 0, PLF_ARRAY_CAPACITY_BYTES);
		}
	}



	PLF_CONSTFUNC void reset(const size_type index)
	{
		if PLF_CONSTEXPR (hardened) check_index_is_within_size(index);

		buffer[index / PLF_TYPE_BITWIDTH] &= ~(storage_type(1) << (index % PLF_TYPE_BITWIDTH));
	}



	PLF_CONSTFUNC void reset_range(const size_type begin, const size_type end)
	{
		if PLF_CONSTEXPR (hardened)
		{
			check_index_is_within_size(begin);
			check_index_is_within_size(end);
		}

		if (begin == end)
		#ifdef PLF_CPP20_SUPPORT
			[[unlikely]]
		#endif
		{
			return;
		}

		const size_type begin_type_index = begin / PLF_TYPE_BITWIDTH, end_type_index = (end - 1) / PLF_TYPE_BITWIDTH, begin_subindex = begin % PLF_TYPE_BITWIDTH, distance_to_end_storage = PLF_TYPE_BITWIDTH - (end % PLF_TYPE_BITWIDTH);

		if (begin_type_index != end_type_index)
		{
			buffer[begin_type_index] &= ~(std::numeric_limits<storage_type>::max() << begin_subindex);

			#ifdef PLF_CONSTEVAL_SUPPORT
				if consteval
				{
					std::fill_n(buffer + begin_type_index + 1, (end_type_index - 1) - begin_type_index, 0);
				}
				else
			#endif
			{
				std::memset(plf::void_cast(buffer + begin_type_index + 1), 0, ((end_type_index - 1) - begin_type_index) * sizeof(storage_type));
			}

			buffer[end_type_index] &= ~(std::numeric_limits<storage_type>::max() >> distance_to_end_storage);
		}
		else
		{
			buffer[begin_type_index] &= ~((std::numeric_limits<storage_type>::max() << begin_subindex) & (std::numeric_limits<storage_type>::max() >> distance_to_end_storage));
		}
	}



	PLF_CONSTFUNC void flip() PLF_NOEXCEPT
	{
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) buffer[current] = ~buffer[current];
		set_overflow_to_zero();
	}



	PLF_CONSTFUNC void flip(const size_type index)
	{
		buffer[index / PLF_TYPE_BITWIDTH] ^= storage_type(1) << (index % PLF_TYPE_BITWIDTH);
	}



	PLF_CONSTFUNC bool all() const PLF_NOEXCEPT
	{
		return all_words_equal<static_cast<storage_type>(~storage_type())>(buffer, buffer + (PLF_ARRAY_CAPACITY - 1)) && last_word_with_overflow_set() == std::numeric_limits<storage_type>::max();
	}



	PLF_CONSTFUNC bool all_range(const size_type begin, const size_type end) const
	{
		if PLF_CONSTEXPR (hardened)
		{
			check_index_is_within_size(begin);
			check_index_is_within_size(end);
		}

		if (begin == end)
		#ifdef PLF_CPP20_SUPPORT
			[[unlikely]]
		#endif
		{
			return false;
		}

		const size_type begin_type_index = begin / PLF_TYPE_BITWIDTH, end_type_index = (end - 1) / PLF_TYPE_BITWIDTH, begin_subindex = begin % PLF_TYPE_BITWIDTH, distance_to_end_storage = PLF_TYPE_BITWIDTH - (end % PLF_TYPE_BITWIDTH);

		if (begin_type_index != end_type_index) // ie. if first and last bit to be set are not in the same storage_type unit
		{
			// Check first storage_type:
			if ((buffer[begin_type_index] | ~(std::numeric_limits<storage_type>::max() << begin_subindex)) != std::numeric_limits<storage_type>::max())
			{
				return false;
			}

			// Check all intermediate storage_type's (if any):
			if (!all_words_equal<static_cast<storage_type>(~storage_type())>(buffer + begin_type_index + 1, buffer + end_type_index))
			{
				return false;
			}

			// Check last storage_type:
			if ((buffer[end_type_index] | ~(std::numeric_limits<storage_type>::max() >> distance_to_end_storage)) != std::numeric_limits<storage_type>::max())
			{
				return false;
			}
		}
		else
		{
			if ((buffer[begin_type_index] | ~((std::numeric_limits<storage_type>::max() << begin_subindex) & (std::numeric_limits<storage_type>::max() >> distance_to_end_storage))) != std::numeric_limits<storage_type>::max())
			{
				return false;
			}
		}

		return true;
	}



	PLF_CONSTFUNC bool any() const PLF_NOEXCEPT
	{
		return buffer[0] != 0 || !all_words_equal<0>(buffer + 1, buffer + PLF_ARRAY_CAPACITY); // See first_one() for why the first storage_type is checked on it's own
	}



	PLF_CONSTFUNC bool any_range(const size_type begin, const size_type end) const
	{
		if PLF_CONSTEXPR (hardened)
		{
			check_index_is_within_size(begin);
			check_index_is_within_size(end);
		}

		if (begin == end)
		#ifdef PLF_CPP20_SUPPORT
			[[unlikely]]
		#endif
		{
			return false;
		}

		const size_type begin_type_index = begin / PLF_TYPE_BITWIDTH, end_type_index = (end - 1) / PLF_TYPE_BITWIDTH, begin_subindex = begin % PLF_TYPE_BITWIDTH, distance_to_end_storage = PLF_TYPE_BITWIDTH - (end % PLF_TYPE_BITWIDTH);

		if (begin_type_index != end_type_index)
		{
			if ((buffer[begin_type_index] & (std::numeric_limits<storage_type>::max() << begin_subindex)) != 0) return true;

			if (!all_words_equal<0>(buffer + begin_type_index + 1, buffer + end_type_index)) return true;

			if ((buffer[end_type_index] & (std::numeric_limits<storage_type>::max() >> distance_to_end_storage)) != 0) return true;
		}
		else
		{
			if ((buffer[begin_type_index] & ((std::numeric_limits<storage_type>::max() << begin_subindex) & (std::numeric_limits<storage_type>::max() >> distance_to_end_storage))) != 0) return true;
		}

		return false;
	}



	PLF_CONSTFUNC bool none() const PLF_NOEXCEPT

	{
		return !any();
	}



	PLF_CONSTFUNC bool none_range(const size_type begin, const size_type end) const
	{
		return !any_range(begin, end);
	}



	PLF_CONSTFUNC size_type count() const PLF_NOEXCEPT
	{
		size_type total = 0;

		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current)
		{
			total += plf::popcount(buffer[current]);
		}

		return total;
	}



	PLF_CONSTFUNC size_type count_range(const size_type begin, const size_type end) const
	{
		if PLF_CONSTEXPR (hardened)
		{
			check_index_is_within_size(begin);
			check_index_is_within_size(end);
		}

		if (begin == end)
		#ifdef PLF_CPP20_SUPPORT
			[[unlikely]]
		#endif
		{
			return 0;
		}

		const size_type begin_type_index = begin / PLF_TYPE_BITWIDTH, end_type_index = (end - 1) / PLF_TYPE_BITWIDTH, begin_subindex = begin % PLF_TYPE_BITWIDTH, distance_to_end_storage = PLF_TYPE_BITWIDTH - (end % PLF_TYPE_BITWIDTH);

		if (begin_type_index != end_type_index) // ie. if first and last bit to be set are not in the same storage_type unit
		{
			// Count first storage_type:
			size_type total = plf::popcount(buffer[begin_type_index] & (std::numeric_limits<storage_type>::max() << begin_subindex));

			// Count all intermediate storage_type's (if any):
			for (size_type current = begin_type_index + 1; current != end_type_index; ++current)
			{
				total += plf::popcount(buffer[current]);
			}

			// Count last storage_type:
			total += plf::popcount(buffer[end_type_index] & (std::numeric_limits<storage_type>::max() >> distance_to_end_storage));
			return total;
		}
		else
		{
			return plf::popcount(buffer[begin_type_index] & ((std::numeric_limits<storage_type>::max() << begin_subindex) & (std::numeric_limits<storage_type>::max() >> distance_to_end_storage)));
		}
	}



private:


	PLF_CONSTFUNC size_type search_one_forwards(const size_type word_index) const PLF_NOEXCEPT
	{
		const storage_type * const found = find_word_forwards<0>(buffer + word_index, buffer + PLF_ARRAY_CAPACITY);
		return (found == buffer + PLF_ARRAY_CAPACITY) ? std::numeric_limits<size_type>::max() : (static_cast<size_type>(found - buffer) * PLF_TYPE_BITWIDTH) + plf::countr_zero(*found);
	}



	PLF_CONSTFUNC size_type search_one_backwards(const size_type word_index) const PLF_NOEXCEPT
	{
		const storage_type * const found = find_word_backwards<0>(buffer, buffer + word_index + 1);
		return (found == NULL) ? std::numeric_limits<size_type>::max() : (((static_cast<size_type>(found - buffer) + 1) * PLF_TYPE_BITWIDTH) - plf::countl_zero(*found)) - 1;
	}



	PLF_CONSTFUNC size_type search_zero_forwards(const size_type word_index) const PLF_NOEXCEPT
	{ // Overflow bits are always 0, so a zero found at or past total_size means there are no zeroes within the bitset
		const storage_type * const found = find_word_forwards<static_cast<storage_type>(~storage_type())>(buffer + word_index, buffer + PLF_ARRAY_CAPACITY);
		if (found == buffer + PLF_ARRAY_CAPACITY) return std::numeric_limits<size_type>::max();

		const size_type index = (static_cast<size_type>(found - buffer) * PLF_TYPE_BITWIDTH) + plf::countr_one(*found);
		return (index < total_size) ? index : std::numeric_limits<size_type>::max();
	}



	PLF_CONSTFUNC size_type search_zero_backwards(const size_type word_index) const PLF_NOEXCEPT
	{ // Must not be called on the final storage_type, as it doesn't account for overflow bits
		const storage_type * const found = find_word_backwards<static_cast<storage_type>(~storage_type())>(buffer, buffer + word_index + 1);
		return (found == NULL) ? std::numeric_limits<size_type>::max() : (((static_cast<size_type>(found - buffer) + 1) * PLF_TYPE_BITWIDTH) - plf::countl_one(*found)) - 1;
	}



public:

	PLF_CONSTFUNC size_type first_one() const PLF_NOEXCEPT
	{ // Check the first storage_type on it's own, as the search's 8-at-a-time skipping is slower when the result is in the first storage_type
		if (buffer[0] != 0) return plf::countr_zero(buffer[0]);
		return search_one_forwards(1);
	}



	PLF_CONSTFUNC size_type next_one(size_type index) const PLF_NOEXCEPT
	{
		if (index >= total_size) return std::numeric_limits<size_type>::max();

		size_type word_index = index / PLF_TYPE_BITWIDTH;
		index %= PLF_TYPE_BITWIDTH; // convert to sub-index within word
		const storage_type current_word = buffer[word_index] >> index;

		if (index != PLF_TYPE_BITWIDTH && current_word != 0) // Note: shifting by full bitwidth of type is undefined behaviour, so can't rely on word << 64 being zero
		{
			return (word_index * PLF_TYPE_BITWIDTH) + plf::countr_zero(current_word) + index;
		}

		if (++word_index == PLF_ARRAY_CAPACITY) return std::numeric_limits<size_type>::max();

		return search_one_forwards(word_index);
	}



	PLF_CONSTFUNC size_type last_one() const PLF_NOEXCEPT
	{
		const storage_type last_word = buffer[PLF_ARRAY_CAPACITY - 1];

		if (last_word != 0) return ((PLF_ARRAY_CAPACITY_BITS - plf::countl_zero(last_word)) - 1);
		if (PLF_ARRAY_CAPACITY == 1) return std::numeric_limits<size_type>::max();
		return search_one_backwards(PLF_ARRAY_CAPACITY - 2);
	}



	PLF_CONSTFUNC size_type prev_one(size_type index) const PLF_NOEXCEPT
	{
		if (index >= total_size) return std::numeric_limits<size_type>::max();

		const size_type word_index = index / PLF_TYPE_BITWIDTH;
		index %= PLF_TYPE_BITWIDTH;

		const storage_type current_word = buffer[word_index] << (PLF_TYPE_BITWIDTH - index);

		if (index != 0 && current_word != 0)
		{
			return ((word_index * PLF_TYPE_BITWIDTH) + index - 1) - plf::countl_zero(current_word);
		}

		if (word_index == 0) return std::numeric_limits<size_type>::max();

		return search_one_backwards(word_index - 1);
	}



	PLF_CONSTFUNC size_type first_zero() const PLF_NOEXCEPT
	{ // See first_one() for why the first storage_type is checked on it's own
		if (buffer[0] != std::numeric_limits<storage_type>::max())
		{
			const size_type index = plf::countr_one(buffer[0]);
			return (index < total_size) ? index : std::numeric_limits<size_type>::max();
		}

		return search_zero_forwards(1);
	}



	PLF_CONSTFUNC size_type next_zero(size_type index) const PLF_NOEXCEPT
	{
		if (index >= total_size) return std::numeric_limits<size_type>::max();

		size_type word_index = index / PLF_TYPE_BITWIDTH;
		index %= PLF_TYPE_BITWIDTH;
		const storage_type current_word = buffer[word_index] | ~(std::numeric_limits<storage_type>::max() << index);

		if (current_word != std::numeric_limits<storage_type>::max())
		{
			index = (word_index * PLF_TYPE_BITWIDTH) + plf::countr_one(current_word);
			return (index < total_size) ? index : std::numeric_limits<size_type>::max();
		}

		if (++word_index == PLF_ARRAY_CAPACITY) return std::numeric_limits<size_type>::max();

		return search_zero_forwards(word_index);
	}



	PLF_CONSTFUNC size_type last_zero() const PLF_NOEXCEPT
	{
		const storage_type last_word = last_word_with_overflow_set();

		if (last_word != std::numeric_limits<storage_type>::max()) return ((PLF_ARRAY_CAPACITY_BITS - plf::countl_one(last_word)) - 1);
		if (PLF_ARRAY_CAPACITY == 1) return std::numeric_limits<size_type>::max();
		return search_zero_backwards(PLF_ARRAY_CAPACITY - 2);
	}



	PLF_CONSTFUNC size_type prev_zero(size_type index) const PLF_NOEXCEPT
	{
		if (index >= total_size) return std::numeric_limits<size_type>::max();

		const size_type word_index = index / PLF_TYPE_BITWIDTH;
		index %= PLF_TYPE_BITWIDTH;

		const storage_type current_word = buffer[word_index] | ~(std::numeric_limits<storage_type>::max() >> ((PLF_TYPE_BITWIDTH - 1) - index));

		if (current_word != std::numeric_limits<storage_type>::max())
		{
			return (((word_index + 1) * PLF_TYPE_BITWIDTH) - plf::countl_one(current_word)) - 1;
		}

		if (word_index == 0) return std::numeric_limits<size_type>::max();

		return search_zero_backwards(word_index - 1);
	}



	PLF_CONSTFUNC void operator = (const bitset &source) PLF_NOEXCEPT
	{
		std::copy(source.buffer, source.buffer + PLF_ARRAY_CAPACITY, buffer);
	}



 	PLF_CONSTFUNC bool operator == (const bitset &source) const PLF_NOEXCEPT
	{
		return std::equal(source.buffer, source.buffer + PLF_ARRAY_CAPACITY, buffer);
	}



 	PLF_CONSTFUNC bool operator != (const bitset &source) const PLF_NOEXCEPT
	{
		return !(*this == source);
	}



	PLF_CONSTFUNC size_type size() const PLF_NOEXCEPT
 	{
 		return total_size;
 	}



	PLF_CONSTFUNC size_type memory() const PLF_NOEXCEPT
 	{
 		return sizeof(*this) + PLF_ARRAY_CAPACITY_BYTES;
 	}



	PLF_CONSTFUNC bitset & operator &= (const bitset& source) PLF_NOEXCEPT
	{
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) buffer[current] &= source.buffer[current];
		return *this;
	}



	PLF_CONSTFUNC bitset operator & (const bitset& source) const PLF_NOEXCEPT
	{
		bitset result;
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) result.buffer[current] = buffer[current] & source.buffer[current];
		return result;
	}



	PLF_CONSTFUNC bitset & operator |= (const bitset& source) PLF_NOEXCEPT
	{
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) buffer[current] |= source.buffer[current];
		return *this;
	}



	PLF_CONSTFUNC bitset operator | (const bitset& source) const PLF_NOEXCEPT
	{
		bitset result;
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) result.buffer[current] = buffer[current] | source.buffer[current];
		return result;
	}



	PLF_CONSTFUNC bitset & operator ^= (const bitset& source) PLF_NOEXCEPT
	{
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) buffer[current] ^= source.buffer[current];
		return *this;
	}



	PLF_CONSTFUNC bitset operator ^ (const bitset& source) const PLF_NOEXCEPT
	{
		bitset result;
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) result.buffer[current] = buffer[current] ^ source.buffer[current];
		return result;
	}



	PLF_CONSTFUNC bitset operator ~ () const
	{
		bitset<total_size, storage_type> temp(*this);
		temp.flip();
		return temp;
	}



	PLF_CONSTFUNC bitset & operator >>= (size_type shift_amount) PLF_NOEXCEPT
	{
		size_type end = PLF_ARRAY_CAPACITY - 1;

		if (shift_amount >= PLF_TYPE_BITWIDTH)
		{
			size_type current = 0;

			if (shift_amount < total_size)
			#ifdef PLF_CPP20_SUPPORT
				[[likely]]
			#endif
			{
				size_type current_source = shift_amount / PLF_TYPE_BITWIDTH;

				if ((shift_amount %= PLF_TYPE_BITWIDTH) != 0)
				{
					const storage_type shifter = static_cast<storage_type>(PLF_TYPE_BITWIDTH - shift_amount);

					for (; current_source != end; ++current, ++current_source)
					{
						buffer[current] = (buffer[current_source] >> shift_amount) | (buffer[current_source + 1] << shifter);
					}

					buffer[current++] = buffer[end] >> shift_amount;
				}
				else
				{
					++end;

					for (; current_source != end; ++current, ++current_source)
					{
						buffer[current] = buffer[current_source];
					}
				}
			}

			#ifdef PLF_CONSTEVAL_SUPPORT
				if consteval
				{
					std::fill_n(buffer + current, PLF_ARRAY_CAPACITY - current, 0);
				}
				else
			#endif
			{
				std::memset(plf::void_cast(buffer + current), 0, (PLF_ARRAY_CAPACITY - current) * sizeof(storage_type));
			}
		}
		else if (shift_amount != 0)
		{
			const storage_type shifter = static_cast<storage_type>(PLF_TYPE_BITWIDTH - shift_amount);

			for (size_type current = 0; current != end; ++current)
			{
				buffer[current] = (buffer[current] >> shift_amount) | (buffer[current + 1] << shifter);
			}

			buffer[end] >>= shift_amount;
		}

		return *this;
	}



	PLF_CONSTFUNC bitset operator >> (const size_type shift_amount) const
	{
		bitset<total_size, storage_type> temp(*this);
		return temp >>= shift_amount;
	}



	// >>= but only from a given index onwards
	PLF_CONSTFUNC void shift_left_range (size_type shift_amount, const size_type first) PLF_NOEXCEPT
	{
		assert(first < total_size);

		size_type end = PLF_ARRAY_CAPACITY - 1;
		const size_type first_word_index = first / PLF_TYPE_BITWIDTH;
		const storage_type first_word = buffer[first_word_index];

		if (shift_amount >= PLF_TYPE_BITWIDTH)
		{
			size_type current = first_word_index;

			if (shift_amount < total_size - first)
			#ifdef PLF_CPP20_SUPPORT
				[[likely]]
			#endif
			{
				size_type current_source = first_word_index + (shift_amount / PLF_TYPE_BITWIDTH);

				if ((shift_amount %= PLF_TYPE_BITWIDTH) != 0)
				{
					const storage_type shifter = static_cast<storage_type>(PLF_TYPE_BITWIDTH - shift_amount);

					for (; current_source != end; ++current, ++current_source)
					{
						buffer[current] = (buffer[current_source] >> shift_amount) | (buffer[current_source + 1] << shifter);
					}

					buffer[current++] = buffer[end] >> shift_amount;
				}
				else
				{
					++end;

					for (; current_source != end; ++current, ++current_source)
					{
						buffer[current] = buffer[current_source];
					}
				}
			}

			#ifdef PLF_CONSTEVAL_SUPPORT
				if consteval
				{
					std::fill_n(buffer + current, PLF_ARRAY_CAPACITY - current, 0);
				}
				else
			#endif
			{
				std::memset(plf::void_cast(buffer + current), 0, (PLF_ARRAY_CAPACITY - current) * sizeof(storage_type));
			}
		}
		else if (shift_amount != 0)
		{
			const storage_type shifter = static_cast<storage_type>(PLF_TYPE_BITWIDTH - shift_amount);

			for (size_type current = first_word_index; current != end; ++current)
			{
				buffer[current] = (buffer[current] >> shift_amount) | (buffer[current + 1] << shifter);
			}

			buffer[end] >>= shift_amount;
		}

		// Restore X bits to first word
		const storage_type remainder = static_cast<storage_type>(first - (first_word_index * PLF_TYPE_BITWIDTH));
  		buffer[first_word_index] = (buffer[first_word_index] & (std::numeric_limits<storage_type>::max() << remainder)) | (first_word & (std::numeric_limits<storage_type>::max() >> (PLF_TYPE_BITWIDTH - remainder)));
	}



	// An optimization of the above for shifting by 1:
	PLF_CONSTFUNC void shift_left_range_one (const size_type first) PLF_NOEXCEPT
	{
		assert(first < total_size);

		const size_type end = PLF_ARRAY_CAPACITY - 1, first_word_index = first / PLF_TYPE_BITWIDTH;
		const storage_type first_word = buffer[first_word_index], shifter = PLF_TYPE_BITWIDTH - 1;

		for (size_type current = first_word_index; current != end; ++current)
		{
			buffer[current] = (buffer[current] >> 1) | (buffer[current + 1] << shifter);
		}

		buffer[end] >>= 1;

		// Restore X bits to first word
		const storage_type remainder = static_cast<storage_type>(first - (first_word_index * PLF_TYPE_BITWIDTH));
  		buffer[first_word_index] = (buffer[first_word_index] & (std::numeric_limits<storage_type>::max() << remainder)) | (first_word & (std::numeric_limits<storage_type>::max() >> (PLF_TYPE_BITWIDTH - remainder)));
	}



	PLF_CONSTFUNC bitset & operator <<= (size_type shift_amount) PLF_NOEXCEPT
	{
		size_type current = PLF_ARRAY_CAPACITY;

		if (shift_amount < total_size)
		#ifdef PLF_CPP20_SUPPORT
			[[likely]]
		#endif
		{
			size_type current_source = PLF_ARRAY_CAPACITY - (shift_amount / PLF_TYPE_BITWIDTH);

			if ((shift_amount %= PLF_TYPE_BITWIDTH) != 0)
			{
				const storage_type shifter = static_cast<storage_type>(PLF_TYPE_BITWIDTH - shift_amount);

				while (--current_source != 0)
				{
					buffer[--current] = (buffer[current_source - 1] >> shifter) | (buffer[current_source] << shift_amount);
				}

				buffer[--current] = buffer[current_source] << shift_amount;
			}
			else
			{
				do
				{
					buffer[--current] = buffer[--current_source];
				} while (current_source != 0);
			}
		}

		#ifdef PLF_CONSTEVAL_SUPPORT
			if consteval
			{
				std::fill_n(buffer, current, 0);
			}
			else
		#endif
		{
			std::memset(plf::void_cast(buffer), 0, current * sizeof(storage_type));
		}

		set_overflow_to_zero();
		return *this;
	}



	PLF_CONSTFUNC bitset operator << (const size_type shift_amount) const
	{
		bitset<total_size, storage_type> temp(*this);
		return temp <<= shift_amount;
	}



	#ifdef PLF_CPP11_SUPPORT
		template <class char_type = char, class traits = std::char_traits<char_type>, class string_allocator_type = std::allocator<char_type> >
		PLF_CONSTFUNC std::basic_string<char_type, traits, string_allocator_type> to_string(const char_type zero = char_type('0'), char_type one = char_type('1')) const
		{
			std::basic_string<char_type, traits, string_allocator_type> temp(total_size, zero);
	#else
		PLF_CONSTFUNC std::basic_string<char> to_string(const char zero = char('0'), char one = char('1')) const
		{
			std::basic_string<char> temp(total_size, zero);
	#endif
		one -= zero;

		for (size_type index = 0, end = PLF_ARRAY_CAPACITY; index != end; ++index)
		{
 			if (buffer[index] != 0)
 			{
 				const size_type string_index = index * PLF_TYPE_BITWIDTH;
 				const storage_type value = buffer[index];

				#ifdef PLF_CPP20_SUPPORT // Avoid the branch otherwise
					if constexpr (total_size % PLF_TYPE_BITWIDTH == 0)
					{
						for (storage_type subindex = 0, sub_end = PLF_TYPE_BITWIDTH; subindex != sub_end; ++subindex)
						{
							temp[total_size - (string_index + subindex + 1)] = zero + (((value >> subindex) & storage_type(1)) * one);
						}
					}
					else
				#endif
				{
					for (storage_type subindex = 0, sub_end = PLF_TYPE_BITWIDTH; subindex != sub_end && (string_index + subindex) != total_size; ++subindex)
					{
						temp[total_size - (string_index + subindex + 1)] = zero + (((value >> subindex) & storage_type(1)) * one);
					}
				}
			}
		}

		return temp;
	}




	#ifdef PLF_CPP11_SUPPORT
		template <class char_type = char, class traits = std::char_traits<char_type>, class string_allocator_type = std::allocator<char_type> >
		PLF_CONSTFUNC std::basic_string<char_type, traits, string_allocator_type> to_rstring(const char_type zero = char_type('0'), char_type one = char_type('1')) const
		{
			std::basic_string<char_type, traits, string_allocator_type> temp(total_size, zero);
	#else
		PLF_CONSTFUNC std::basic_string<char> to_rstring(const char zero = char('0'), char one = char('1')) const
		{
			std::basic_string<char> temp(total_size, zero);
	#endif
		one -= zero;

		for (size_type index = 0, end = PLF_ARRAY_CAPACITY; index != end; ++index)
		{
 			if (buffer[index] != 0)
 			{
 				const size_type string_index = index * PLF_TYPE_BITWIDTH;
 				const storage_type value = buffer[index];

				#ifdef PLF_CPP20_SUPPORT
					if constexpr (total_size % PLF_TYPE_BITWIDTH == 0)
					{
						for (storage_type subindex = 0, sub_end = PLF_TYPE_BITWIDTH; subindex != sub_end; ++subindex)
						{
							temp[string_index + subindex] = zero + (((value >> subindex) & storage_type(1)) * one);
						}
					}
					else
				#endif
				{
					for (storage_type subindex = 0, sub_end = PLF_TYPE_BITWIDTH; subindex != sub_end && (string_index + subindex) != total_size; ++subindex)
					{
						temp[string_index + subindex] = zero + (((value >> subindex) & storage_type(1)) * one);
					}
				}
			}
		}

		return temp;
	}




private:

	template <typename number_type>
	PLF_CONSTFUNC void check_bitset_representable() const
	{
		if (total_size > static_cast<size_type>(std::log10(static_cast<double>(std::numeric_limits<number_type>::max()))) + 1)
		{
			#ifdef PLF_EXCEPTIONS_SUPPORT
				throw std::overflow_error("Bitset cannot be represented by this type due to the size of the bitset");
			#else
				std::terminate();
			#endif
		}
	}


	template <typename number_type>
	PLF_CONSTFUNC number_type to_type() const
	{
		check_bitset_representable<number_type>();
		number_type value = 0;

		for (size_type index = 0, multiplier = 1; index != total_size; ++index, multiplier *= 10)
		{
			value += static_cast<number_type>(operator [](index) * multiplier);
		}

		return value;
	}



	template <typename number_type>
	PLF_CONSTFUNC number_type to_reverse_type() const
	{
		check_bitset_representable<number_type>();
		number_type value = 0;

		for (size_type reverse_index = total_size, multiplier = 1; reverse_index != 0; multiplier *= 10)
		{
			value += operator [](--reverse_index) * multiplier;
		}

		return value;
	}



public:

	PLF_CONSTFUNC unsigned long to_ulong() const
	{
		return to_type<unsigned long>();
	}



	PLF_CONSTFUNC unsigned long to_reverse_ulong() const
	{
		return to_reverse_type<unsigned long>();
	}



	#ifdef PLF_CPP11_SUPPORT
		PLF_CONSTFUNC unsigned long long to_ullong() const
		{
			return to_type<unsigned long long>();
		}



		PLF_CONSTFUNC unsigned long long to_rullong() const
		{
			return to_reverse_type<unsigned long long>();
		}
	#endif


	PLF_CONSTFUNC void swap(bitset &source) PLF_NOEXCEPT
	{
		for (size_type current = 0, end = PLF_ARRAY_CAPACITY; current != end; ++current) std::swap(buffer[current], source.buffer[current]);
	}
};


} // plf namespace


namespace std
{

	template<std::size_t total_size, typename storage_type, bool hardened>
	PLF_CONSTFUNC void swap (plf::bitset<total_size, storage_type, hardened> &a, plf::bitset<total_size, storage_type, hardened> &b) PLF_NOEXCEPT
	{
		a.swap(b);
	}


	template<std::size_t total_size, typename storage_type, bool hardened>
	PLF_CONSTFUNC ostream& operator << (ostream &os, const plf::bitset<total_size, storage_type, hardened> &bs)
	{
		return os << bs.to_string();
	}
}


#undef PLF_TYPE_BITWIDTH
#undef PLF_ARRAY_CAPACITY
#undef PLF_ARRAY_CAPACITY_BITS
#undef PLF_ARRAY_CAPACITY_BYTES
#undef PLF_NO_VECTORIZE

#ifdef PLF_BITSET_DEFINES
	#include "plf_tools_undef.h"
#endif

#endif // PLF_BITSET_H
