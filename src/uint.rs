// These lint errors come from `uint::construct_uint` and can't be disabled on item level.
#![warn(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]

use std::{fmt, marker, iter::{Sum, Step}, convert::TryFrom};
use std::ops::{Sub, Add, AddAssign, Shl, ShlAssign, ShrAssign, Range};
use num_traits::{Zero, One};
use uint::{
	construct_uint, overflowing, panic_on_overflow,
	uint_overflowing_add, uint_overflowing_add_reg,
	uint_overflowing_sub, uint_overflowing_sub_reg,
	uint_overflowing_mul, uint_overflowing_mul_reg,
	uint_overflowing_binop, uint_full_mul_reg, unroll,
	impl_mul_from, impl_mulassign_from, impl_map_from,
	impl_std_for_uint, impl_heapsize_for_uint, impl_quickcheck_arbitrary_for_uint,
};
use rayon::{iter::plumbing, prelude::{ParallelIterator, IndexedParallelIterator}};
use serde_derive::{Serialize, Deserialize};

use crate::{CacheIndex16, CacheIndex32, CacheIndex64, CacheIndex128, CacheIndex192, CacheIndex256};

pub trait MinMax {
	type BoundsType: Copy;
	const MIN: Self::BoundsType;
	const MAX: Self::BoundsType;
}

#[derive(Debug)]
pub struct TryFromIntError<S, T> {
	value: S,
	phantom: marker::PhantomData<T>,
}

impl<S, T> TryFromIntError<S, T> {
	pub fn new(value: S) -> Self {
		TryFromIntError {
			value,
			phantom: marker::PhantomData
		}
	}
}

impl<S: fmt::Display, T: MinMax> fmt::Display for TryFromIntError<S, T>
	where T::BoundsType: fmt::Display
{
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		write!(
			fmt,
			"attempted conversion of {}, out of range: [{}, {}] for given type",
			self.value,
			T::MIN,
			T::MAX,
		)
	}
}

impl<S: fmt::Debug + fmt::Display, T: fmt::Debug + MinMax> std::error::Error for TryFromIntError<S, T>
	where T::BoundsType: fmt::Display {}

pub trait SmallUint: Copy + Zero + MinMax + Sub<u8, Output = Self> + TryFrom<u8> {
	fn is_one(&self) -> bool;
}

macro_rules! construct_small_uint {
	($t: ident, $next: ty, $k: ident, $i: ty, $i_next: ty) => {
		#[derive(Debug, Clone, Copy, PartialEq)]
		pub struct $t(u8);

		impl From<TryFromIntError<$next, $i_next>> for $t {
			fn from(e: TryFromIntError<$next, $i_next>) -> Self {
				$t(u8::from(e.value))
			}
		}

		impl From<$i> for $t {
			fn from(value: $i) -> Self {
				$t(u8::from(value))
			}
		}

		impl TryFrom<u8> for $t {
			type Error = TryFromIntError<u8, $t>;

			fn try_from(value: u8) -> Result<Self, Self::Error> {
				if value <= $t::MAX {
					Ok($t(value))
				} else {
					Err(TryFromIntError::<u8, $t>::new(value))
				}
			}
		}

		impl Sub<u8> for $t {
			type Output = Self;
			fn sub(self, rhs: u8) -> Self { $t(self.0.sub(rhs))}
		}

		impl Add for $t {
			type Output = Self;
			fn add(self, rhs: Self) -> Self { $t(self.0.add(rhs.0))}
		}

		impl Zero for $t {
			fn zero() -> Self { $t(0) }
			fn is_zero(&self) -> bool { self.0 == 0 }
		}

		impl MinMax for $t {
			type BoundsType = u8;
			const MIN: u8 = 0;
			const MAX: u8 = ($k::NUMBER_OF_BITS - 1) as _;
		}

		impl SmallUint for $t {
			fn is_one(&self) -> bool { self.0 == 1 }
		}

		impl From<$t> for u8 {
			fn from(value: $t) -> Self { value.0 }
		}
	}
}
construct_small_uint!(U4, U5, u16, CacheIndex16, CacheIndex32);
construct_small_uint!(U5, U6, u32, CacheIndex32, CacheIndex64);
construct_small_uint!(U6, U7, u64, CacheIndex64, CacheIndex128);
construct_small_uint!(U7, U7Half, u128, CacheIndex128, CacheIndex192);
construct_small_uint!(U7Half, u8, U192, CacheIndex192, CacheIndex256);

impl MinMax for u8 {
	type BoundsType = u8;
	const MIN: u8 = u8::min_value();
	const MAX: u8 = u8::max_value();
}

impl SmallUint for u8 {
	fn is_one(&self) -> bool { One::is_one(self) }
}

// Copy of `rayon::range::IterProducer`, we need it to implement
// `rayon::iter::plumbing::UnindexedProducer` for `IterProducer<large uint>`.
struct IterProducer<T> { range: Range<T> }

impl<T> IntoIterator for IterProducer<T> where Range<T>: Iterator {
	type Item = <Range<T> as Iterator>::Item;
	type IntoIter = Range<T>;

	fn into_iter(self) -> Self::IntoIter { self.range }
}

// Copy of `rayon::range::Iter`, we need it to implement
// `rayon::iter::ParallelIterator` for `Iter<large uint>`.
pub struct Iter<T> { range: Range<T> }

// Copy of `rayon::iter::IntoParallelIterator`, we need it to have unified
// `into_par_iter` on `Range<large uint>` and `Range<primitive integer>`.
pub trait IntoParallelIterator {
	type Item: Send;
	type Iter: ParallelIterator<Item = Self::Item>;
	fn into_par_iter(self) -> Self::Iter;
}

// For types for which rayon has `IntoParallelIterator` implementation
// just delegate our trait's types and method to rayon's.
macro_rules! into_par_iter_delegate {
	($t: ty) => {
		impl IntoParallelIterator for Range<$t> {
			type Item = <Self as rayon::iter::IntoParallelIterator>::Item;
			type Iter = <Self as rayon::iter::IntoParallelIterator>::Iter;

			fn into_par_iter(self) -> Self::Iter {
				rayon::iter::IntoParallelIterator::into_par_iter(self)
			}
		}
	}
}
into_par_iter_delegate!(u8);
into_par_iter_delegate!(u16);
into_par_iter_delegate!(u32);
into_par_iter_delegate!(u64);
into_par_iter_delegate!(u128);

macro_rules! try_from_for_u64_or_less {
	($s: ident, $($t: ident),*) => {$(
		impl TryFrom<$s> for $t {
			type Error = TryFromIntError<$s, $t>;

			#[allow(clippy::cast_lossless)] // Using `as` instead of `from` to cover `usize`.
			fn try_from(u: $s) -> Result<$t, Self::Error> {
				for x in u.0.iter().skip(1) {
					if *x != 0 { return Err(TryFromIntError::<$s, $t>::new(u)); }
				}
				if u.0[0] > std::$t::MAX as u64 {
					Err(TryFromIntError::<$s, $t>::new(u))
				} else {
					Ok(u.0[0] as $t)
				}
			}
		}
	)*}
}

macro_rules! construct_large_uint {
	($t: ident, $i: tt) => {
		construct_uint! {
			#[derive(Serialize, Deserialize)]
			pub struct $t($i);
		}

		impl Zero for $t {
			fn zero() -> Self { Self::zero() }
			fn is_zero(&self) -> bool { self.is_zero() }
		}

		impl One for $t {
			fn one() -> Self { Self::one() }
		}

		impl Sum for $t {
			fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::zero(), Add::add)
			}
		}

		impl Step for $t {
			fn steps_between(start: &Self, end: &Self) -> Option<usize> {
				usize::try_from(end.saturating_sub(*start)).ok()
			}
		
			fn add_usize(&self, n: usize) -> Option<Self> {
				self.checked_add($t::from(n))
			}
		
			fn replace_one(&mut self) -> Self {
				std::mem::replace(self, Self::one())
			}
		
			fn replace_zero(&mut self) -> Self {
				std::mem::replace(self, Self::zero())
			}
		
			fn add_one(&self) -> Self { Add::add(*self, 1) }
		
			fn sub_one(&self) -> Self { Sub::sub(*self, 1) }
		}

		impl plumbing::UnindexedProducer for IterProducer<$t> {
			type Item = $t;
		
			fn split(mut self) -> (Self, Option<Self>) {
				let index = self.range.end.saturating_sub(self.range.start) / 2;
				if index.is_zero() {
					(self, None)
				} else {
					let mid = self.range.start + index;
					let right = mid .. self.range.end;
					self.range.end = mid;
					(self, Some(IterProducer { range: right }))
				}
			}
		
			fn fold_with<F>(self, folder: F) -> F
				where F: plumbing::Folder<Self::Item>
			{
				folder.consume_iter(self)
			}
		}

		impl ParallelIterator for Iter<$t> {
			type Item = $t;
		
			fn drive_unindexed<C>(self, consumer: C) -> C::Result
				where C: plumbing::UnindexedConsumer<Self::Item>
			{
				if let Some(len) = self.opt_len() {
					rayon::iter::IntoParallelIterator::into_par_iter(0..len)
						.map(|i| self.range.start + i)
						.drive(consumer)
				} else {
					plumbing::bridge_unindexed(
						IterProducer { range: self.range },
						consumer
					)
				}
			}
		
			fn opt_len(&self) -> Option<usize> {
				$t::steps_between(&self.range.start, &self.range.end)
			}
		}

		impl IntoParallelIterator for Range<$t> {
			type Item = <Iter<$t> as ParallelIterator>::Item;
			type Iter = Iter<$t>;
		
			fn into_par_iter(self) -> Self::Iter { Iter { range: self } }
		}

		try_from_for_u64_or_less!($t, usize, u64, u32, u16, u8);

		impl TryFrom<$t> for u128 {
			type Error = TryFromIntError<$t, u128>;

			fn try_from(u: $t) -> Result<u128, Self::Error> {
				for x in u.0.iter().skip(2) {
					if *x != 0 { return Err(TryFromIntError::<$t, u128>::new(u)); }
				}
				Ok(u.low_u128())
			}
		}
	}
}
construct_large_uint!(U192, 3);
construct_large_uint!(U256, 4);

impl From<U192> for U256 {
	fn from(value: U192) -> Self {
		Self([value.0[0], value.0[1], value.0[2], 0])
	}
}

impl TryFrom<U256> for U192 {
	type Error = TryFromIntError<U256, U192>;

	fn try_from(u: U256) -> Result<U192, Self::Error> {
		if u.0[3] == 0 {
			Ok(u.convert())
		} else {
			Err(TryFromIntError::<U256, U192>::new(u))
		}
	}
}

// Possibly lossy conversion
pub trait UintConvert<U> { fn convert(self) -> U; }

macro_rules! impl_uint_convert {
	($t: ty) => {
		impl UintConvert<u8> for $t { fn convert(self) -> u8 { self as u8 } }
		#[allow(clippy::cast_lossless)]
		impl UintConvert<u16> for $t { fn convert(self) -> u16 { self as u16 } }
		#[allow(clippy::cast_lossless)]
		impl UintConvert<u32> for $t { fn convert(self) -> u32 { self as u32 } }
		#[allow(clippy::cast_lossless)]
		impl UintConvert<u64> for $t { fn convert(self) -> u64 { self as u64 } }
		#[allow(clippy::cast_lossless)]
		impl UintConvert<u128> for $t { fn convert(self) -> u128 { self as u128 } }
		impl UintConvert<U192> for $t { fn convert(self) -> U192 { U192::from(self) } }
		impl UintConvert<U256> for $t { fn convert(self) -> U256 { U256::from(self) } }
	}
}
impl_uint_convert!(u8);
impl_uint_convert!(u16);
impl_uint_convert!(u32);
impl_uint_convert!(u64);
impl_uint_convert!(u128);

macro_rules! impl_uint_convert_large {
	($t: ty) => {
		impl UintConvert<u8> for $t { fn convert(self) -> u8 { self.low_u64() as u8 } }
		impl UintConvert<u16> for $t { fn convert(self) -> u16 { self.low_u64() as u16 } }
		impl UintConvert<u32> for $t { fn convert(self) -> u32 { self.low_u32() } }
		impl UintConvert<u64> for $t { fn convert(self) -> u64 { self.low_u64() } }
		impl UintConvert<u128> for $t { fn convert(self) -> u128 { self.low_u128() } }
	}
}
impl_uint_convert_large!(U192);
impl_uint_convert_large!(U256);

impl UintConvert<U192> for U192 { fn convert(self) -> U192 { self } }
impl UintConvert<U256> for U192 { fn convert(self) -> U256 { U256::from(self) } }

impl UintConvert<U192> for U256 {
	fn convert(self) -> U192 { U192([self.0[0], self.0[1], self.0[2]]) }
}
impl UintConvert<U256> for U256 { fn convert(self) -> U256 { self } }

pub trait Uint: Copy + Zero + One + AddAssign + ShlAssign<u8> + ShrAssign<u8> + Shl<u8, Output = Self> {
	const NUMBER_OF_BITS: u32;
	fn trailing_zeros(self) -> u32;
	fn leading_zeros(self) -> u32;
}

macro_rules! impl_uint {
	($t: ty, $bn: expr) => {
		impl Uint for $t {
			const NUMBER_OF_BITS: u32 = $bn;
			fn trailing_zeros(self) -> u32 { self.trailing_zeros() }
			fn leading_zeros(self) -> u32 { self.leading_zeros() }
		}
	}
}
impl_uint!(u8, 8);
impl_uint!(u16, 16);
impl_uint!(u32, 32);
impl_uint!(u64, 64);
impl_uint!(u128, 128);

macro_rules! impl_uint_large {
	($t: ty, $bn: expr) => {
		impl Uint for $t {
			const NUMBER_OF_BITS: u32 = $bn;
			fn trailing_zeros(self) -> u32 { (&self).trailing_zeros() }
			fn leading_zeros(self) -> u32 { (&self).leading_zeros() }
		}
	}
}
impl_uint_large!(U192, 192);
impl_uint_large!(U256, 256);
