use std::{sync, convert::TryFrom as _};
use num_traits::Zero;
use rayon::prelude::ParallelSliceMut as _;
use serde_derive::{Serialize, Deserialize};

use crate::uint::{U192, U256};

#[derive(Debug, PartialEq, Clone, Default, Serialize, Deserialize)]
struct UintMapBase<K: Ord, V> {
	vec: Vec<(K, V)>,
	sorted: bool,
}

impl <K: Ord + Send, V: Copy + Send> UintMapBase<K, V> {
	fn get_uint(&self, k: &K) -> Option<V> {
		if self.sorted {
			self.vec.binary_search_by_key(&k, |&(ref key, _)| key)
				.map(|i| self.vec[i].1)
				.ok()
		} else {
			for (key, val) in self.vec.iter() {
				if k == key {
					return Some(*val);
				}
			}
			None
		}
	}

	fn insert_uint(&mut self, k: K, v: V) {
		self.sorted = false;
		self.vec.push((k, v));
	}

	fn optimize(&mut self) {
		self.vec.shrink_to_fit();
		self.vec.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
		self.sorted = true;
	}
}

pub trait UintMap {
	type K;

	fn get_uint(&self, k: &Self::K) -> Option<Self::K>;
	fn insert_uint(&mut self, k: Self::K, v: Self::K);
	fn optimize(&mut self);
}

macro_rules! construct_uint_map {
	($m: ident { $first: ident : $v_first: ident, $($field: ident : $v: ident,)* @ $last: ident : $k: ident }) => {
		#[derive(Debug, PartialEq, Clone, Default, Serialize, Deserialize)]
		pub struct $m {
			$first: UintMapBase<$k, $v_first>,
			$($field: UintMapBase<$k, $v>,)*
			$last: UintMapBase<$k, $k>,
		}

		impl UintMap for $m {
			type K = $k;

			fn get_uint(&self, k: &$k) -> Option<$k> {
				self.$first.get_uint(k).map($k::from)
				$(
					.or_else(|| self.$field.get_uint(k).map($k::from))
				)*
					.or_else(|| self.$last.get_uint(k))
			}

			fn insert_uint(&mut self, k: $k, v: $k) {
				if let Ok(vv) = $v_first::try_from(v) {
					self.$first.insert_uint(k, vv)
				} else
				$(
					if let Ok(vv) = $v::try_from(v) {
						self.$field.insert_uint(k, vv)
					} else
				)*
				{
					self.$last.insert_uint(k, v)
				}
			}

			fn optimize(&mut self) {
				self.$first.optimize();
				$(self.$field.optimize();)+
				self.$last.optimize();
			}
		}
	}
}
construct_uint_map!(UintMap32 { map_8: u8, map_16: u16, @ map_32: u32 });
construct_uint_map!(UintMap64 { map_8: u8, map_16: u16, map_32: u32, @ map_64: u64 });
construct_uint_map!(UintMap128 { map_8: u8, map_16: u16, map_32: u32, map_64: u64, @ map_128: u128 });
construct_uint_map!(UintMap192 { map_8: u8, map_16: u16, map_32: u32, map_64: u64, map_128: u128, @ map_192: U192 });
construct_uint_map!(UintMap256 { map_8: u8, map_16: u16, map_32: u32, map_64: u64, map_128: u128, map_192: U192, @ map_256: U256 });

pub struct UintMapMutex<'a, M: UintMap>(sync::Mutex<&'a mut M>);

impl<'a, M> UintMapMutex<'a, M> where M: UintMap, M::K: Zero {
	pub fn with_map(map: &'a mut M) -> Self {
		UintMapMutex(sync::Mutex::new(map))
	}

	pub fn into_inner(self) -> std::sync::LockResult<&'a mut M> {
		self.0.into_inner()
	}

	pub fn insert_non_zero(&self, k: M::K, v: M::K) {
		if !v.is_zero() {
			self.0.lock().unwrap().insert_uint(k, v);
		}
	}
}
