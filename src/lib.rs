#![feature(
	vec_resize_default, step_trait, associated_type_defaults, result_map_or_else,
	proc_macro_hygiene
)]

use std::{ops, fmt, fs, path, convert::TryFrom, iter::{Sum, Step}};
use num_traits::{Zero as _, One as _};
use rayon::prelude::ParallelIterator as _;

mod rem;

mod uint;
use crate::uint::{
	TryFromIntError, MinMax, U4, U5, U6, U7, U7Half, U192, SmallUint, Uint,
	UintConvert as _, IntoParallelIterator
};
pub use crate::uint::U256;

mod uint_map;
use crate::uint_map::{UintMap, UintMap32, UintMap64, UintMap128, UintMap192, UintMap256, UintMapMutex};

trait CacheIndex: Sized + Copy + Step + MinMax<BoundsType = u8> {
	type K: Uint + CacheTypes<U=Self::U, I=Self>;
	type U: SmallUint + From<Self>;
	type UPrev;

	const NUMBER: usize = Self::K::NUMBER_OF_BITS as usize - Self::MIN as usize;

	fn intersection_range(min: u8, max: u8) -> ops::RangeInclusive<Self>;
	fn to_index(&self) -> usize;
}

macro_rules! construct_cache_index {
	($i: ident, $u: ty, $u_prev: ty, $k: ident, $min: expr) => {
		#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
		struct $i(u8);

		impl Step for $i {
			fn steps_between(start: &Self, end: &Self) -> Option<usize> {
				u8::steps_between(&start.0, &end.0)
			}

			fn add_usize(&self, n: usize) -> Option<Self> {
				self.0.add_usize(n).map($i)
			}

			fn replace_one(&mut self) -> Self {
				std::mem::replace(self, $i(1))
			}

			fn replace_zero(&mut self) -> Self {
				std::mem::replace(self, $i(0))
			}

			fn add_one(&self) -> Self {
				$i(self.0.add_one())
			}

			fn sub_one(&self) -> Self {
				$i(self.0.sub_one())
			}
		}

		impl MinMax for $i {
			type BoundsType = u8;
			const MIN: u8 = $min as _;
			const MAX: u8 = ($k::NUMBER_OF_BITS - 1) as _;
		}

		impl CacheIndex for $i {
			type K = $k;
			type U = $u;
			type UPrev = $u_prev;

			fn intersection_range(min: u8, max: u8) -> ops::RangeInclusive<Self> {
				$i(min.max(Self::MIN))..=$i(max.min(Self::MAX))
			}

			fn to_index(&self) -> usize { (self.0 - Self::MIN) as usize }
		}

		impl From<$i> for u8 {
			fn from(value: $i) -> Self { value.0 }
		}
	}
}

construct_cache_index!(CacheIndex16, U4, (), u16, 0);
construct_cache_index!(CacheIndex32, U5, U4, u32, u16::NUMBER_OF_BITS);
construct_cache_index!(CacheIndex64, U6, U5, u64, u32::NUMBER_OF_BITS);
construct_cache_index!(CacheIndex128, U7, U6, u128, u64::NUMBER_OF_BITS);
construct_cache_index!(CacheIndex192, U7Half, U7, U192, u128::NUMBER_OF_BITS);
construct_cache_index!(CacheIndex256, u8, U7Half, U256, U192::NUMBER_OF_BITS);

impl From<U4> for CacheIndex16 {
	fn from(value: U4) -> Self {
		CacheIndex16(u8::from(value))
	}
}

impl From<TryFromIntError<U5, CacheIndex32>> for CacheIndex16 {
	fn from(e: TryFromIntError<U5, CacheIndex32>) -> Self {
		CacheIndex16::from(U4::from(e))
	}
}

impl TryFrom<u8> for CacheIndex16 {
	type Error = TryFromIntError<u8, CacheIndex16>;

	fn try_from(value: u8) -> Result<Self, Self::Error> {
		if value <= Self::MAX {
			Ok(Self(value))
		} else {
			Err(TryFromIntError::<u8, CacheIndex16>::new(value))
		}
	}
}

macro_rules! impl_try_from {
	($i: ident, $u: ident, $us: ident) => {
		impl TryFrom<$u> for $i {
			type Error = TryFromIntError<$u, $i>;

			fn try_from(value: $u) -> Result<Self, Self::Error> {
				let n = u8::from(value);
				if n >= Self::MIN {
					Ok($i(n))
				} else {
					Err(TryFromIntError::<$u, $i>::new(value))
				}
			}
		}

		impl TryFrom<u8> for $i {
			type Error = TryFromIntError<u8, $i>;

			fn try_from(value: u8) -> Result<Self, Self::Error> {
				if Self::MIN <= value && value <= Self::MAX {
					Ok($i(value))
				} else {
					Err(TryFromIntError::<u8, $i>::new(value))
				}
			}
		}
	}
}

impl_try_from!(CacheIndex32, U5, U4);
impl_try_from!(CacheIndex64, U6, U5);
impl_try_from!(CacheIndex128, U7, U6);
impl_try_from!(CacheIndex192, U7Half, U7);

impl TryFrom<u8> for CacheIndex256 {
	type Error = TryFromIntError<u8, CacheIndex256>;

	fn try_from(value: u8) -> Result<Self, Self::Error> {
		if value >= Self::MIN {
			Ok(Self(value))
		} else {
			Err(TryFromIntError::<u8, CacheIndex256>::new(value))
		}
	}
}

trait CacheTypes: Uint {
	type U: SmallUint + From<Self::I>;
	type UPrev;
	type I: CacheIndex<K=Self, U=Self::U>;
}

macro_rules! impl_cache_types {
	($k: ty, $u: ty, $i: ty) => {
		impl CacheTypes for $k {
			type U = $u;
			type UPrev = <Self::I as CacheIndex>::UPrev;
			type I = $i;
		}
	}
}

impl_cache_types!(u16, U4, CacheIndex16);
impl_cache_types!(u32, U5, CacheIndex32);
impl_cache_types!(u64, U6, CacheIndex64);
impl_cache_types!(u128, U7, CacheIndex128);
impl_cache_types!(U192, U7Half, CacheIndex192);
impl_cache_types!(U256, u8, CacheIndex256);

fn reduce<K: Uint + CacheTypes>(n: K::U, a_rank: K, k: u8) -> (Option<K::U>, K) {
	let mut n = n;

	let mut a_rank = a_rank;
	let mut b_rank = K::one() << (k - 1);
	if a_rank.is_zero() { return (Some(n), b_rank); }

	let mut a_last = a_rank.trailing_zeros() as u8 + 1;
	let mut b_last = k;

	loop {
		if a_last < b_last {
			std::mem::swap(&mut a_rank, &mut b_rank);
			std::mem::swap(&mut a_last, &mut b_last);
		} else if a_last == b_last {
			return (
				if n.is_one() { Some(K::U::zero()) } else { None },
				K::zero()
			);
		} else {
			let diff = a_last - b_last;
			// Calculation  of `b_last % diff` was a bottleneck,
			// so it's replaced with precalculated `rem::rem(b_last, diff)`.
			let rem = rem::rem(b_last, diff);

			n = n - b_last;

			a_rank >>= a_last - 1;
			a_last = diff - rem;
			if rem != 0 {
				a_rank <<= rem;
				a_rank += K::one();
			}
			a_rank <<= a_last - 1;

			b_rank >>= b_last;
			if b_rank.is_zero() { return (Some(n), a_rank); }
			b_last = b_rank.trailing_zeros() as u8 + 1;
		}
	}
}

fn cache_segment_16_key(n: CacheIndex16, a_rank: u16) -> usize {
	((1u16 << u8::from(n)) + a_rank) as _
}

#[derive(Debug, Clone, Default)]
struct CacheSegmentSlice16<'a> {
	data: &'a[u16]
}

impl<'a> CacheSegmentSlice16<'a> {
	fn get_or_zero(&self, n: CacheIndex16, a_rank: u16) -> u16 {
		self.data[cache_segment_16_key(n, a_rank)]
	}
}

#[derive(Debug, Clone, Default)]
struct CacheSegmentSlice<'a, M: UintMap> where M::K: CacheTypes {
	data: &'a[(u8, M)]
}

impl<'a, M: UintMap> CacheSegmentSlice<'a, M> where M::K: CacheTypes {
	fn get_or_zero(&self, n: <M::K as CacheTypes>::I, a_rank: M::K) -> M::K {
		if let Some((_, n_cache)) = self.data.get(n.to_index()) {
			if let Some(result) = n_cache.get_uint(&a_rank) {
				return result;
			}
		}

		M::K::zero()
	}
}

const CACHE_SEGMENT_16_LEN: usize = 65_536;

#[derive(Clone)]
struct CacheSegment16 { data: [u16; CACHE_SEGMENT_16_LEN] }

impl<'a> CacheSegment16 {
	fn as_slice(&self) -> CacheSegmentSlice16 {
		CacheSegmentSlice16 { data: &self.data[..] }
	}

	fn get_or_zero(&self, n: CacheIndex16, a_rank: u16) -> u16 {
		self.as_slice().get_or_zero(n, a_rank)
	}

	fn insert(&mut self, n: CacheIndex16, a_rank: u16, val: u16) {
		self.data[cache_segment_16_key(n, a_rank)] = val;
	}

	fn init(&mut self) { self.data[1] = 1 }

	fn clear(&mut self) {
		for x in self.data.iter_mut() { *x = 0 }
		self.init()
	}
}

impl fmt::Debug for CacheSegment16 {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Debug::fmt(&&self.data[..], f)
	}
}

impl PartialEq<CacheSegment16> for CacheSegment16 {
	fn eq(&self, other: &CacheSegment16) -> bool { self.data[..] == other.data[..] }
}

impl Default for CacheSegment16 {
	fn default() -> Self {
		let mut res = Self { data: [0; CACHE_SEGMENT_16_LEN] };
		res.init();
		res
	}
}

// TODO: When Const Generics arrive Use single generic struct `CacheSegment<M>
// instead of separate `CacheSegment...` types and `CacheSegment` trait.
trait CacheSegment where <Self::M as UintMap>::K: CacheTypes {
	type M: UintMap + Default;

	fn as_slice(&self) -> CacheSegmentSlice<Self::M>;
	fn get_or_zero(&self, n: <<Self::M as UintMap>::K as CacheTypes>::I, a_rank: <Self::M as UintMap>::K) -> <Self::M as UintMap>::K;
	fn insert(&mut self, n: <<Self::M as UintMap>::K as CacheTypes>::I, val: (u8, Self::M));
	fn get_pre_slice_and_mut(&mut self, n: <<Self::M as UintMap>::K as CacheTypes>::I) -> (CacheSegmentSlice<Self::M>, &mut (u8, Self::M));
	fn clear(&mut self);
}

macro_rules! construct_cache_segment {
	($cs: ident, $m: ty) => {
		#[derive(Clone)]
		struct $cs {
			data: [(u8, $m); <<$m as UintMap>::K as CacheTypes>::I::NUMBER]
		}

		impl CacheSegment for $cs {
			type M = $m;

			fn as_slice(&self) -> CacheSegmentSlice<Self::M> {
				CacheSegmentSlice { data: &self.data[..] }
			}

			fn get_or_zero(&self, n: <<Self::M as UintMap>::K as CacheTypes>::I, a_rank: <Self::M as UintMap>::K) -> <Self::M as UintMap>::K {
				self.as_slice().get_or_zero(n, a_rank)
			}

			fn insert(&mut self, n: <<Self::M as UintMap>::K as CacheTypes>::I, val: (u8, Self::M)) {
				self.data[n.to_index()] = val;
			}

			fn get_pre_slice_and_mut(&mut self, n: <<Self::M as UintMap>::K as CacheTypes>::I) -> (CacheSegmentSlice<Self::M>, &mut (u8, Self::M)) {
				let n = n.to_index();
				debug_assert!(n < self.data.len());
				let (left, right) = self.data.split_at_mut(n);
				(CacheSegmentSlice { data: left }, right.first_mut().unwrap())
			}

			fn clear(&mut self) {
				self.data.iter_mut().for_each(|x| *x = <(u8, Self::M)>::default());
			}
		}

		impl fmt::Debug for $cs {
			fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
				fmt::Debug::fmt(&&self.data[..], f)
			}
		}

		impl PartialEq<$cs> for $cs {
			fn eq(&self, other: &$cs) -> bool { self.data[..] == other.data[..] }
		}

		impl Default for $cs {
			fn default() -> Self {
				Self {
					data: array_macro::array![
						<(u8, $m)>::default();
						<<$m as UintMap>::K as CacheTypes>::I::NUMBER
					]
				}
			}
		}
	}
}

construct_cache_segment!(CacheSegment32, UintMap32);
construct_cache_segment!(CacheSegment64, UintMap64);
construct_cache_segment!(CacheSegment128, UintMap128);
construct_cache_segment!(CacheSegment192, UintMap192);
construct_cache_segment!(CacheSegment256, UintMap256);

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Cache {
	path: Option<path::PathBuf>,
	n: u8,
	cache_16: CacheSegment16,
	cache_32: CacheSegment32,
	cache_64: CacheSegment64,
	cache_128: CacheSegment128,
	cache_192: CacheSegment192,
	cache_256: CacheSegment256,
}

#[derive(Debug, Default)]
struct CacheSlice<'a> {
	cache_16: CacheSegmentSlice16<'a>,
	cache_32: CacheSegmentSlice<'a, UintMap32>,
	cache_64: CacheSegmentSlice<'a, UintMap64>,
	cache_128: CacheSegmentSlice<'a, UintMap128>,
	cache_192: CacheSegmentSlice<'a, UintMap192>,
	cache_256: CacheSegmentSlice<'a, UintMap256>,
}

trait CacheMut<I: CacheIndex> where I::K: CacheTypes {
	type M: UintMap<K=I::K>;

	fn insert(&mut self, n: I, val: (u8, Self::M));
	fn insert_from(&mut self, u: I::U, f: fs::File) -> bincode::Result<()>;
	fn pre_compute_nn_u(&mut self, u: I::U, n: u8) -> bincode::Result<()>;
	fn get_pre_slice_and_mut(&mut self, n: I) -> (CacheSlice, &mut (u8, Self::M));
}

impl CacheMut<CacheIndex32> for Cache {
	type M = UintMap32;

	fn insert(&mut self, n: CacheIndex32, val: (u8, Self::M)) {
		self.cache_32.insert(n, val)
	}

	fn insert_from(&mut self, u: U5, f: fs::File) -> bincode::Result<()> {
		if let Ok(i) = CacheIndex32::try_from(u) {
			self.insert(i, bincode::deserialize_from(f)?)
		}
		
		Ok(())
	}

	fn pre_compute_nn_u(&mut self, u: U5, n: u8) -> bincode::Result<()> {
		match CacheIndex32::try_from(u) {
			Ok(i) => self.pre_compute_nn(i, n)?,
			Err(e) => self.pre_compute_small_nn(CacheIndex16::from(e)),
		}
		self.n += 1;
		Ok(())
	}

	fn get_pre_slice_and_mut(&mut self, n: CacheIndex32) -> (CacheSlice, &mut (u8, Self::M)) {
		let (cache_slice, n_cache) = self.cache_32.get_pre_slice_and_mut(n);
		(
			CacheSlice {
				cache_16: self.cache_16.as_slice(),
				cache_32: cache_slice,
				..Default::default()
			},
			n_cache,
		)
	}
}

macro_rules! impl_cache_mut {
	($i: ident, $i_prev: ty, $m: ty, $($full_field: ident,)* @ $field: ident) => {
		impl CacheMut<$i> for Cache {
			type M = $m;

			fn insert(&mut self, n: $i, val: (u8, $m)) {
				self.$field.insert(n, val)
			}

			fn insert_from(&mut self, u: <$i as CacheIndex>::U, f: fs::File) -> bincode::Result<()> {
 				match $i::try_from(u) {
 					Ok(i) => Ok(self.insert(i, bincode::deserialize_from(f)?)),
 					Err(e) => CacheMut::<$i_prev>::insert_from(self, <$i as CacheIndex>::UPrev::from(e), f),
 				}
			}

			fn pre_compute_nn_u(&mut self, u: <$i as CacheIndex>::U, n: u8) -> bincode::Result<()> {
 				match $i::try_from(u) {
 					Ok(i) => {
 						self.pre_compute_nn(i, n)?;
 						self.n += 1;
 					},
 					Err(e) => CacheMut::<$i_prev>::pre_compute_nn_u(self, <$i as CacheIndex>::UPrev::from(e), n)?,
 				};
				Ok(())
			}

			fn get_pre_slice_and_mut(&mut self, n: $i) -> (CacheSlice, &mut (u8, $m)) {
				let (cache_slice, n_cache) = self.$field.get_pre_slice_and_mut(n);
				(
					#[allow(clippy::needless_update)]
					CacheSlice {
						$(
							$full_field: self.$full_field.as_slice(),
						)*
						$field: cache_slice,
						..Default::default()
					},
					n_cache,
				)
			}
		}
	}
}
impl_cache_mut!(CacheIndex64, CacheIndex32, UintMap64, cache_16, cache_32, @ cache_64);
impl_cache_mut!(CacheIndex128, CacheIndex64, UintMap128, cache_16, cache_32, cache_64, @ cache_128);
impl_cache_mut!(CacheIndex192, CacheIndex128, UintMap192, cache_16, cache_32, cache_64, cache_128, @ cache_192);
impl_cache_mut!(CacheIndex256, CacheIndex192, UintMap256, cache_16, cache_32, cache_64, cache_128, cache_192, @ cache_256);

trait CacheGet<K: Uint + CacheTypes> where K::U: From<K::I>, u8: From<K::I> {
	fn get_or_zero(&self, n: K::U, a_rank: K) -> K;

	fn cache_get(&self, n: Option<K::U>, a_rank: K) -> K {
		n.map_or(K::zero(), |n| self.get_or_zero(n, a_rank))
	}

	fn calculate(&self, n: K::I, a_rank: K) -> K {
		let mut sum = K::zero();
		for k in 1..=u8::from(n) {
			let (n, a_rank) = reduce(K::U::from(n), a_rank, k);
			if let Some(n) = n { sum += self.get_or_zero(n, a_rank) }
		}
		sum
	}
}

impl CacheGet<u16> for Cache {
	fn get_or_zero(&self, n: <u16 as CacheTypes>::U, a_rank: u16) -> u16 {
		self.cache_16.get_or_zero(n.into(), a_rank)
	}
}

impl<'a> CacheGet<u16> for CacheSlice<'a> {
	fn get_or_zero(&self, n: <u16 as CacheTypes>::U, a_rank: u16) -> u16 {
		self.cache_16.get_or_zero(n.into(), a_rank)
	}
}

macro_rules! impl_cache_get {
	($k: ident, $ks: ty, $field: ident) => {
		impl CacheGet<$k> for Cache {
			fn get_or_zero(&self, n: <$k as CacheTypes>::U, a_rank: $k) -> $k {
 				match <$k as CacheTypes>::I::try_from(n) {
 					Ok(i) => self.$field.get_or_zero(i, a_rank),
 					Err(e) => $k::from(<Self as CacheGet<$ks>>::get_or_zero(self, <$k as CacheTypes>::UPrev::from(e), a_rank.convert()))
 				}
			}
		}

		impl<'a> CacheGet<$k> for CacheSlice<'a> {
			fn get_or_zero(&self, n: <$k as CacheTypes>::U, a_rank: $k) -> $k {
 				match <$k as CacheTypes>::I::try_from(n) {
 					Ok(i) => self.$field.get_or_zero(i, a_rank),
 					Err(e) => $k::from(<Self as CacheGet<$ks>>::get_or_zero(self, <$k as CacheTypes>::UPrev::from(e), a_rank.convert()))
 				}
			}
		}
	}
}
impl_cache_get!(u32, u16, cache_32);
impl_cache_get!(u64, u32, cache_64);
impl_cache_get!(u128, u64, cache_128);
impl_cache_get!(U192, u128, cache_192);
impl_cache_get!(U256, U192, cache_256);

impl Cache {
	pub fn new<P: AsRef<path::Path>>(path: P) -> bincode::Result<Self> {
		let mut cache = Self::default();

		fs::create_dir_all(&path)?;
		let mut max_nn = 0;
		for entry in fs::read_dir(&path)? {
			let entry = entry?;
			if let Some(file_name) = entry.file_name().to_str() {
				if let Ok(nn) = u8::from_str_radix(file_name, 10) {
					let f_path = &entry.path();
					let f = fs::File::open(f_path)?;
					CacheMut::<CacheIndex256>::insert_from(&mut cache, nn, f)?;
					
					if max_nn < nn { max_nn = nn }

					log::info!("Data nn={} read from \"{}\" file.",
						nn, f_path.display()
					);
				}
			}
		}
		
		if 15 < max_nn { cache.pre_compute_small(15) }

		cache.path = Some(path::PathBuf::from(path.as_ref()));
		Ok(cache)
	}

	pub fn clear(&mut self) {
		self.n = 0;
		self.cache_16.clear();
		self.cache_32.clear();
		self.cache_64.clear();
		self.cache_128.clear();
		self.cache_192.clear();
		self.cache_256.clear();
	}

	fn min_nn(old_n: u8) -> u8 { old_n / 2 + old_n % 2 + 1 }

	fn pre_compute_small_nn(&mut self, nn_ind: CacheIndex16) {
		let nn = u8::from(nn_ind);

		// TODO: Calculate only necessary ones for given `n` as in `pre_compute_nn`?
		// It would be useful only for `n` up to 30, so is it worth the trouble?
		for a_rank in 1..(1u16 << nn) {
			let sum = self.calculate(nn_ind, a_rank);
			self.cache_16.insert(nn_ind, a_rank, sum);
		}
		let sum = self.calculate(nn_ind, 0);
		self.cache_16.insert(nn_ind, 0, sum);

		log::debug!("Data for nn={} precomputed.", nn);
	}

	fn pre_compute_small(&mut self, n: u8) {
		for nn_ind in CacheIndex16::intersection_range(self.n + 1, n) {
			self.pre_compute_small_nn(nn_ind)
		}
	}

	fn pre_compute_nn<I>(&mut self, nn_ind: I, n: u8) -> bincode::Result<()>
		where I: CacheIndex + Sync + Send,
		      u8: From<I>,
		      I::K: Ord + Uint + Sum + CacheTypes<U=I::U, I=I> + Sync + Send + fmt::Display,
		      Self: CacheMut<I>,
		      <Self as CacheMut<I>>::M: Send + Sync + serde::Serialize,
		      for<'a> CacheSlice<'a>: CacheGet<I::K> + Send + Sync,
		      ops::Range<I::K>: IntoParallelIterator<Item=I::K>
	{
		let nn = u8::from(nn_ind);
		let path = self.path.clone();
		let (ref cache_slice, (old_n, n_cache)) = self.get_pre_slice_and_mut(nn_ind);
		if n <= *old_n || nn < Self::min_nn(*old_n) { return Ok(()); }
		let n_cache = UintMapMutex::with_map(n_cache);

		let min_tot = old_n.saturating_sub(nn);
		let max_tot = (n - nn).min(nn);
		let a_rank_base_min = I::K::one() << min_tot;
		let a_rank_base_max = I::K::one() << max_tot;
		(a_rank_base_min..a_rank_base_max).into_par_iter().for_each(|a_rank_base| {
			let a_rank = (a_rank_base << 1) + I::K::one();
			for last in 0..(nn - (I::K::NUMBER_OF_BITS - a_rank_base.leading_zeros()) as u8) {
				let a_rank = a_rank << last;
				let sum = cache_slice.calculate(nn_ind, a_rank);
				n_cache.insert_non_zero(a_rank, sum);
			}
		});

		if *old_n < nn {
			let sum: I::K = (0..nn).into_par_iter().map(|last| {
				let a_rank = I::K::one() << last;
				// TODO: Use, here, variant of `calculate` that doesn't call reduce at all?
				let sum = cache_slice.calculate(nn_ind, a_rank);
				n_cache.insert_non_zero(a_rank, sum);
				sum
			}).sum();
			n_cache.insert_non_zero(I::K::zero(), sum);
		}

		let n_cache = n_cache.into_inner().unwrap();
		n_cache.optimize();
		
		*old_n = n;

		log::debug!("Data for n={}, nn={} precomputed.", n, nn);

		if let Some(mut path) = path {
			path.push(nn.to_string());
			let f = fs::File::create(&path)?;
			bincode::serialize_into(f, &(n, n_cache))?;

			log::debug!("Data for n={}, nn={} written into \"{}\" file.",
				n, nn, path.display()
			);
		}

		Ok(())
	}

	fn pre_compute(&mut self, n: u8, max_n: u8) -> bincode::Result<()> {
		if n <= self.n { return Ok(()); }

		self.pre_compute_small(n);

		let min_nn = Self::min_nn(self.n);

		log::info!("Precomputing data for maximal n {}, from {} to {}.",
			max_n, min_nn, n
		);

		for nn in CacheIndex32::intersection_range(min_nn, n) {
			self.pre_compute_nn(nn, max_n)?;
		}
		for nn in CacheIndex64::intersection_range(min_nn, n) {
			self.pre_compute_nn(nn, max_n)?;
		}
		for nn in CacheIndex128::intersection_range(min_nn, n) {
			self.pre_compute_nn(nn, max_n)?;
		}
		for nn in CacheIndex192::intersection_range(min_nn, n) {
			self.pre_compute_nn(nn, max_n)?;
		}
		for nn in CacheIndex256::intersection_range(min_nn, n) {
			self.pre_compute_nn(nn, max_n)?;
		}

		self.n = n;

		Ok(())
	}

	pub fn f(&mut self, n: u8) -> bincode::Result<U256> {
		self.pre_compute(n, n)?;

		Ok(self.cache_get(Some(n), U256::zero()))
	}

	pub fn finite_iter(&mut self, n: u8) -> FiniteIter {
		FiniteIter {
			cache: self,
			n,
			nn: 0
		}
	}
}

pub struct FiniteIter<'a> {
	cache: &'a mut Cache,
	n: u8,
	nn: u8,
}

impl<'a> Iterator for FiniteIter<'a> {
	type Item = bincode::Result<U256>;

	fn next(&mut self) -> Option<Self::Item> {
		let nn = self.nn;
		if nn > self.n { return None; }

		let res = if nn == self.cache.n + 1 {
			CacheMut::<CacheIndex256>::pre_compute_nn_u(self.cache, nn, self.n)
		} else {
			self.cache.pre_compute(nn, self.n)
		};

		if let Err(e) = res { return Some(Err(e)); }

		self.nn += 1;

		Some(Ok(self.cache.cache_get(Some(nn), U256::zero())))
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let len = (self.n - self.nn + 1) as _;
		(len, Some(len))
	}
}

impl<'a> ExactSizeIterator for FiniteIter<'a> {}


#[cfg(test)]
mod tests {
	mod reduce {
		use std::convert::TryFrom;
		use crate::{reduce, U4, U5, U6, U7, U7Half, U192, U256};

		#[test]
		fn zero_empty_one() {
			assert_eq!(
				reduce(U4::try_from(0).unwrap(), 0b0u16, 1),
				(Some(U4::try_from(0).unwrap()), 0b1)
			);
		}

		#[test]
		fn one_empty_one() {
			assert_eq!(
				reduce(U5::try_from(1).unwrap(), 0b0u32, 1),
				(Some(U5::try_from(1).unwrap()), 0b1)
			);
		}
		#[test]
		fn one_one_one() {
			assert_eq!(
				reduce(U6::try_from(1).unwrap(), 0b1u64, 1),
				(Some(U6::try_from(0).unwrap()), 0b0)
			);
		}

		#[test]
		fn two_empty_one() {
			assert_eq!(
				reduce(U7::try_from(2).unwrap(), 0b0u128, 1),
				(Some(U7::try_from(2).unwrap()), 0b1)
			);
		}
		#[test]
		fn two_empty_two() {
			assert_eq!(
				reduce(U4::try_from(2).unwrap(), 0b0u16, 2),
				(Some(U4::try_from(2).unwrap()), 0b10)
			);
		}
		#[test]
		fn two_one_one() {
			assert_eq!(
				reduce(U5::try_from(2).unwrap(), 0b1u32, 1),
				(None, 0b0)
			);
		}
		#[test]
		fn two_one_two() {
			assert_eq!(
				reduce(U6::try_from(2).unwrap(), 0b1u64, 2),
				(Some(U6::try_from(1).unwrap()), 0b1)
			);
		}
		#[test]
		fn two_two_one() {
			assert_eq!(
				reduce(U7::try_from(2).unwrap(), 0b10u128, 1),
				(Some(U7::try_from(1).unwrap()), 0b1)
			);
		}
		#[test]
		fn two_two_two() {
			assert_eq!(
				reduce(U4::try_from(2).unwrap(), 0b10u16, 2),
				(None, 0b0)
			);
		}
		#[test]
		fn two_one_one_one() {
			assert_eq!(
				reduce(U5::try_from(2).unwrap(), 0b11u32, 1),
				(None, 0b0)
			);
		}
		#[test]
		fn two_one_one_two() {
			assert_eq!(
				reduce(U6::try_from(2).unwrap(), 0b11u64, 2),
				(Some(U6::try_from(0).unwrap()), 0b0)
			);
		}

		#[test]
		fn ten_valid_four() {
			assert_eq!(
				reduce(U7::try_from(10).unwrap(), 0b1110_1001_0000_0000u128, 4),
				(Some(U7::try_from(6).unwrap()), 0b1110_1001_0001)
			);
		}
		#[test]
		fn six_valid_three() {
			assert_eq!(
				reduce(U4::try_from(6).unwrap(), 0b1110_1001_0001u16, 3),
				(Some(U4::try_from(3).unwrap()), 0b1_1101_0011)
			);
		}
		#[test]
		fn three_valid_two() {
			assert_eq!(reduce(U5::try_from(3).unwrap(), 0b1_1101_0011u32, 1), (None, 0));
		}

		#[test]
		fn u128_to_none() {
			assert_eq!(
				reduce(U7::try_from(112).unwrap(), 157_925_767_228_541_777_963_483_398_144u128, 47),
				(None, 0)
			);
		}
		#[test]
		fn u128_to_some() {
			assert_eq!(
				reduce(U7::try_from(112).unwrap(), 157_925_767_228_541_777_963_483_398_144u128, 63),
				(Some(U7::try_from(49).unwrap()), 17_122_345_999)
			);
		}

		#[test]
		fn u192_to_none() {
			assert_eq!(
				reduce(
					U7Half::try_from(180).unwrap(),
					U192::from_dec_str("11358496188292163190007198540759695457777614848").unwrap(),
					103
				),
				(None, U192::zero())
			);
		}
		#[test]
		fn u192_to_some() {
			assert_eq!(
				reduce(
					U7Half::try_from(180).unwrap(),
					U192::from_dec_str("11358496188292163190007198540759695457777614848").unwrap(),
					70
				),
				(Some(U7Half::try_from(110).unwrap()), U192::from_dec_str("9621020502745847315826432").unwrap())
			);
		}

		#[test]
		fn u256_to_none() {
			assert_eq!(
				reduce(200, U256::from_dec_str("11481654187554695508963474593350331467988005824225855143936").unwrap(), 147),
				(None, U256::zero())
			);
		}
		#[test]
		fn u256_to_some() {
			assert_eq!(
				reduce(200, U256::from_dec_str("11481654187554695508963474593350331467988005824225855143936").unwrap(), 59),
				(Some(141), U256::from_dec_str("19917495062198544411863819827546043717648").unwrap())
			);
		}
	}

	mod cache {
		use std::convert::TryFrom;
		use std::fmt::Debug;
		use crate::{
			U4, U192, U256, MinMax, Uint, UintMap, CacheMut, Cache, CacheSlice, CacheGet, CacheTypes,
			CacheIndex, CacheIndex16, CacheIndex32, CacheIndex64, CacheIndex128, CacheIndex192, CacheIndex256,
		};

		fn test_insert_get_small<V>(n: CacheIndex16, key: u16, val: V)
			where u16: From<V>
		{
			let val = u16::from(val);
			let n_as_u = U4::from(n);
			let mut cache = Cache::default();
			assert_eq!(cache.get_or_zero(n_as_u, key), 0u16);

			cache.cache_16.insert(n, key, val);
			assert_eq!(cache.get_or_zero(n_as_u, key), val);

			let (slice, _) = cache.get_pre_slice_and_mut(
				CacheIndex32::try_from(u8::from(n) + CacheIndex32::MIN).unwrap()
			);
			assert_eq!(slice.get_or_zero(n_as_u, key), val);
		}

		fn test_insert_get<I, K, V>(n: I, key: K, val: V)
			where I: CacheIndex<K=K, U=K::U>,
			      K: Uint + CacheTypes<I=I> + Debug + From<V> + num_traits::Zero + PartialEq,
			      V: Copy + Debug,
			      K::U: Copy + From<I>,
			      u8: From<I>,
			      Cache: CacheMut<I> + CacheGet<K>,
			      for<'a> CacheSlice<'a>: CacheGet<K>,
		{
			let val = K::from(val);
			let n_as_u = K::U::from(n);
			let mut cache = Cache::default();
			assert_eq!(cache.get_or_zero(n_as_u, key), K::zero());

			let (_, (_, n_cache)) = cache.get_pre_slice_and_mut(n);
			n_cache.insert_uint(key, val);
			assert_eq!(cache.get_or_zero(n_as_u, key), val);

			let (slice, _) = cache.get_pre_slice_and_mut(n.add_one());
			assert_eq!(slice.get_or_zero(n_as_u, key), val);
		}

		#[test]
		fn insert_get_u16_u8() {
			test_insert_get_small(CacheIndex16::try_from(8).unwrap(), 201u16, 16u8)
		}
		#[test]
		fn insert_get_u16_u16() {
			test_insert_get_small(CacheIndex16::try_from(14).unwrap(), 925u16, 567u16)
		}

		#[test]
		fn insert_get_u32_u8() {
			test_insert_get(
				CacheIndex32::try_from(30).unwrap(), 2_984_792_841u32, 255u8
			)
		}
		#[test]
		fn insert_get_u32_u16() {
			test_insert_get(
				CacheIndex32::try_from(25).unwrap(), 4_294_967_295u32, 8183u16
			)
		}
		#[test]
		fn insert_get_u32_u32() {
			test_insert_get(
				CacheIndex32::try_from(16).unwrap(), 65536u32, 417_025u32
			)
		}

		#[test]
		fn insert_get_u64_u8() {
			test_insert_get(
				CacheIndex64::try_from(57).unwrap(), 18_446_744_073_709_551_615u64, 3u8
			)
		}
		#[test]
		fn insert_get_u64_u16() {
			test_insert_get(
				CacheIndex64::try_from(32).unwrap(), 9_823_042_097_502_930u64, 8183u16
			)
		}
		#[test]
		fn insert_get_u64_u32() {
			test_insert_get(
				CacheIndex64::try_from(49).unwrap(), 4_294_967_296u64, 417_025u32
			)
		}
		#[test]
		fn insert_get_u64_u64() {
			test_insert_get(
				CacheIndex64::try_from(62).unwrap(), 57_109_470_129_710u64, 984_710_487_019_247u64
			)
		}

		#[test]
		fn insert_get_u128_u8() {
			test_insert_get(
				CacheIndex128::try_from(73).unwrap(),
				340_282_366_920_938_463_463_374_607_431_768_211_455u128,
				91u8
			)
		}
		#[test]
		fn insert_get_u128_u16() {
			test_insert_get(
				CacheIndex128::try_from(126).unwrap(),
				1_908_410_294_773_982_692_394u128,
				1397u16
			)
		}
		#[test]
		fn insert_get_u128_u32() {
			test_insert_get(
				CacheIndex128::try_from(93).unwrap(),
				18_446_744_073_709_551_616u128,
				264_837u32
			)
		}
		#[test]
		fn insert_get_u128_u64() {
			test_insert_get(
				CacheIndex128::try_from(64).unwrap(),
				5_689_472_398_472_352_936_897u128,
				89_537_694_659_238_562u64
			)
		}
		#[test]
		fn insert_get_u128_u128() {
			test_insert_get(
				CacheIndex128::try_from(111).unwrap(),
				736_487_130_519_869_823_718_264_716u128,
				93_820_387_203_971_308_501_987u128
			)
		}

		#[test]
		fn insert_get_u192_u8() {
			test_insert_get(
				CacheIndex192::try_from(172).unwrap(),
				U192::from_dec_str("567845637846527359283008293570237520737570582730582002").unwrap(),
				121u8
			)
		}
		#[test]
		fn insert_get_u192_u16() {
			test_insert_get(
				CacheIndex192::try_from(128).unwrap(),
				U192::from_dec_str("7462465572285638506830822454302385626180").unwrap(),
				8750u16
			)
		}
		#[test]
		fn insert_get_u192_u32() {
			test_insert_get(
				CacheIndex192::try_from(140).unwrap(),
				U192::from_dec_str("2124142443505485394657826948273563784658383746582").unwrap(),
				1_353_989_063u32
			)
		}
		#[test]
		fn insert_get_u192_u64() {
			test_insert_get(
				CacheIndex192::try_from(190).unwrap(),
				U192::from_dec_str("340282366920938463463374607431768211456").unwrap(),
				353_237_934_814_015u64
			)
		}
		#[test]
		fn insert_get_u192_u128() {
			test_insert_get(
				CacheIndex192::try_from(163).unwrap(),
				U192::from_dec_str("823582323097345698757230956709567857295692380175638").unwrap(),
				556_036_204_124_875_419_149_058_488_322u128
			)
		}
		#[test]
		fn insert_get_u192_u192() {
			test_insert_get(
				CacheIndex192::try_from(151).unwrap(),
				U192::from_dec_str("982423932235203593950829357853045364998234752835097").unwrap(),
				U192::from_dec_str("181768934518262542035230513952420644816885").unwrap()
			)
		}

		#[test]
		fn insert_get_u256_u8() {
			test_insert_get(
				CacheIndex256::try_from(254).unwrap(),
				U256::from_dec_str("5678456378465273592839372851008293570237520737570582730582002").unwrap(),
				153u8
			)
		}
		#[test]
		fn insert_get_u256_u16() {
			test_insert_get(
				CacheIndex256::try_from(217).unwrap(),
				U256::from_dec_str("62771201735386680763830257894232076664161023554444694034512896").unwrap(),
				63255u16
			)
		}
		#[test]
		fn insert_get_u256_u32() {
			test_insert_get(
				CacheIndex256::try_from(201).unwrap(),
				U256::from_dec_str("7896044618650977117854925043439539266349923328202820197287920039565481968").unwrap(),
				467_654_823u32
			)
		}
		#[test]
		fn insert_get_u256_u64() {
			test_insert_get(
				CacheIndex256::try_from(195).unwrap(),
				U256::from_dec_str("6277101735386680763835789423207666416102355444464034512896").unwrap(),
				9_342_735_628_638_756_273u64
			)
		}
		#[test]
		fn insert_get_u256_u128() {
			test_insert_get(
				CacheIndex256::try_from(219).unwrap(),
				U256::from_dec_str("9282358232309791345698757230956037095678572509569213580175638").unwrap(),
				649_720_081_249_801_804_701_410_940u128
			)
		}
		#[test]
		fn insert_get_u256_u192() {
			test_insert_get(
				CacheIndex256::try_from(231).unwrap(),
				U256::from_dec_str("61998242393223520358465939508293578530453649987252347528935097").unwrap(),
				U192::from_dec_str("154835085628553809182760981204917359102931923019281839012").unwrap()
			)
		}
		#[test]
		fn insert_get_u256_u256() {
			test_insert_get(
				CacheIndex256::try_from(192).unwrap(),
				U256::from_dec_str("353315296707596425900855774842552867425807035552911495758213591667270337445").unwrap(),
				U256::from_dec_str("4390540722302298178216846822985902901999500481508529890298111665785095847339").unwrap()
			)
		}
	}

	const F_RESULTS: [crate::U256; 21] = [
		crate::U256([1, 0, 0, 0]),
		crate::U256([1, 0, 0, 0]),
		crate::U256([2, 0, 0, 0]),
		crate::U256([6, 0, 0, 0]),
		crate::U256([14, 0, 0, 0]),
		crate::U256([34, 0, 0, 0]),
		crate::U256([68, 0, 0, 0]),
		crate::U256([150, 0, 0, 0]),
		crate::U256([296, 0, 0, 0]),
		crate::U256([586, 0, 0, 0]),
		crate::U256([1_140, 0, 0, 0]),
		crate::U256([2_182, 0, 0, 0]),
		crate::U256([4_130, 0, 0, 0]),
		crate::U256([7_678, 0, 0, 0]),
		crate::U256([14_368, 0, 0, 0]),
		crate::U256([26_068, 0, 0, 0]),
		crate::U256([48_248, 0, 0, 0]),
		crate::U256([86_572, 0, 0, 0]),
		crate::U256([158_146, 0, 0, 0]),
		crate::U256([281_410, 0, 0, 0]),
		crate::U256([509_442, 0, 0, 0]),
	];

	mod f {
		use crate::{Cache, tests::F_RESULTS};

		#[test]
		fn f_0() { assert_eq!(Cache::default().f(0).unwrap(), F_RESULTS[0]); }
		#[test]
		fn f_1() { assert_eq!(Cache::default().f(1).unwrap(), F_RESULTS[1]); }
		#[test]
		fn f_2() { assert_eq!(Cache::default().f(2).unwrap(), F_RESULTS[2]); }
		#[test]
		fn f_3() { assert_eq!(Cache::default().f(3).unwrap(), F_RESULTS[3]); }
		#[test]
		fn f_4() { assert_eq!(Cache::default().f(4).unwrap(), F_RESULTS[4]); }
		#[test]
		fn f_5() { assert_eq!(Cache::default().f(5).unwrap(), F_RESULTS[5]); }
		#[test]
		fn f_6() { assert_eq!(Cache::default().f(6).unwrap(), F_RESULTS[6]); }
		#[test]
		fn f_7() { assert_eq!(Cache::default().f(7).unwrap(), F_RESULTS[7]); }
		#[test]
		fn f_8() { assert_eq!(Cache::default().f(8).unwrap(), F_RESULTS[8]); }
		#[test]
		fn f_9() { assert_eq!(Cache::default().f(9).unwrap(), F_RESULTS[9]); }
		#[test]
		fn f_10() { assert_eq!(Cache::default().f(10).unwrap(), F_RESULTS[10]); }
		#[test]
		fn f_11() { assert_eq!(Cache::default().f(11).unwrap(), F_RESULTS[11]); }
		#[test]
		fn f_12() { assert_eq!(Cache::default().f(12).unwrap(), F_RESULTS[12]); }
		#[test]
		fn f_13() { assert_eq!(Cache::default().f(13).unwrap(), F_RESULTS[13]); }
		#[test]
		fn f_14() { assert_eq!(Cache::default().f(14).unwrap(), F_RESULTS[14]); }
		#[test]
		fn f_15() {
			let mut cache = Cache::default();
			assert_eq!(cache.f(15).unwrap(), F_RESULTS[15]);
			assert_eq!(cache.cache_16.data[..256], [
				0, 1, 1, 1, 2, 1, 1, 1, 6, 2, 2, 0, 2, 2, 2, 0,
				14, 4, 4, 1, 2, 1, 1, 0, 4, 2, 3, 1, 2, 1, 1, 0,
				34, 8, 10, 0, 5, 3, 2, 0, 5, 0, 3, 0, 0, 0, 0, 0,
				6, 5, 5, 0, 5, 3, 2, 0, 5, 0, 3, 0, 0, 0, 0, 0,
				68, 15, 20, 2, 8, 6, 4, 0, 10, 2, 6, 0, 2, 2, 0, 0,
				3, 2, 2, 0, 3, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0,
				12, 3, 8, 2, 3, 3, 2, 0, 8, 2, 6, 0, 2, 2, 0, 0,
				3, 2, 2, 0, 3, 1, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0,
				150, 28, 44, 0, 19, 11, 8, 0, 24, 0, 14, 0, 0, 0, 0, 0,
				10, 6, 6, 0, 7, 4, 3, 0, 5, 0, 3, 0, 0, 0, 0, 0,
				11, 0, 6, 0, 0, 0, 0, 0, 7, 0, 4, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				14, 11, 10, 0, 12, 7, 5, 0, 12, 0, 7, 0, 0, 0, 0, 0,
				10, 6, 6, 0, 7, 4, 3, 0, 5, 0, 3, 0, 0, 0, 0, 0,
				11, 0, 6, 0, 0, 0, 0, 0, 7, 0, 4, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
			][..]);
		}
		#[test]
		fn f_16() { assert_eq!(Cache::default().f(16).unwrap(), F_RESULTS[16]); }
		#[test]
		fn f_17() { assert_eq!(Cache::default().f(17).unwrap(), F_RESULTS[17]); }
		#[test]
		fn f_18() { assert_eq!(Cache::default().f(18).unwrap(), F_RESULTS[18]); }
		#[test]
		fn f_19() { assert_eq!(Cache::default().f(19).unwrap(), F_RESULTS[19]); }
		#[test]
		fn f_20() { assert_eq!(Cache::default().f(20).unwrap(), F_RESULTS[20]); }

		#[test]
		fn f_7_8_9() {
			let mut c = Cache::default();
			c.f(7).unwrap();
			assert_eq!(c.f(8).unwrap(), F_RESULTS[8]);
			assert_eq!(c.f(9).unwrap(), F_RESULTS[9]);
		}
		#[test]
		fn f_5_10() {
			let mut c = Cache::default();
			c.f(5).unwrap();
			assert_eq!(c.f(10).unwrap(), F_RESULTS[10]);
		}
		#[test]
		fn f_17_18_19() {
			let mut c = Cache::default();
			c.f(17).unwrap();
			assert_eq!(c.f(18).unwrap(), F_RESULTS[18]);
			assert_eq!(c.f(19).unwrap(), F_RESULTS[19]);
		}
		#[test]
		fn f_10_15_20() {
			let mut c = Cache::default();
			c.f(10).unwrap();
			assert_eq!(c.f(15).unwrap(), F_RESULTS[15]);
			assert_eq!(c.f(20).unwrap(), F_RESULTS[20]);
		}
	}

	mod files {
		use crate::{Cache, tests::F_RESULTS};

		fn remove_test_tmp_dir() -> Result<(), failure::Error> {
			use std::fs;

			let test_path = "tmp";
			match fs::read_dir(test_path) {
				Ok(rd) => {
					for entry in rd {
						let entry = entry?;
						let file_name = entry.file_name();
						let file_name = file_name.to_str().unwrap();
						u8::from_str_radix(file_name, 10)?;
						fs::remove_file(entry.path())?;
					}
					fs::remove_dir(test_path)?;
					Ok(())
				},
				Err(e) => match e.kind() {
					std::io::ErrorKind::NotFound => Ok(()),
					_ => Err(e.into())
				}
			}
		}

		#[test]
		fn f_17_18() {
			remove_test_tmp_dir().unwrap();
		
			let mut c1 = Cache::new("tmp").unwrap();
			assert_eq!(c1.f(17).unwrap(), F_RESULTS[17]);

			c1.n = 0;
			let mut c2 = Cache::new("tmp").unwrap();
			assert_eq!(c1, c2);

			assert_eq!(c1.f(18).unwrap(), F_RESULTS[18]);
			assert_eq!(c2.f(18).unwrap(), F_RESULTS[18]);
			assert_eq!(c1, c2);

			remove_test_tmp_dir().unwrap();
		}
	}

	mod iter {
		use crate::{Cache, tests::F_RESULTS};

		#[test]
		fn iter_5() {
			let mut cache = Cache::default();
			let res: Vec<_> = cache.finite_iter(5).map(Result::unwrap).collect();
			assert_eq!(res[..], F_RESULTS[..6])
		}

		#[test]
		fn f_10_iter_17() {
			let mut cache = Cache::default();
			cache.f(10).unwrap();
			let res: Vec<_> = cache.finite_iter(17).map(Result::unwrap).collect();
			assert_eq!(res[..], F_RESULTS[..18])
		}

		#[test]
		fn f_19_iter_18() {
			let mut cache = Cache::default();
			cache.f(19).unwrap();
			let res: Vec<_> = cache.finite_iter(18).map(Result::unwrap).collect();
			assert_eq!(res[..], F_RESULTS[..19])
		}
	}
}
