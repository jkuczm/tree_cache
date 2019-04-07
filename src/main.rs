#![feature(atomic_min_max, duration_float)]

use std::alloc::{System, GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::Instant;

use structopt::StructOpt;

struct Counter;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_MAX: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counter {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		let ret = System.alloc(layout);
		if !ret.is_null() {
			ALLOCATED.fetch_add(layout.size(), SeqCst);
			ALLOCATED_MAX.fetch_max(ALLOCATED.load(SeqCst), SeqCst);
		}
		ret
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		System.dealloc(ptr, layout);
		ALLOCATED.fetch_sub(layout.size(), SeqCst);
	}
}

#[global_allocator]
static A: Counter = Counter;

#[derive(StructOpt)]
struct Cli {
	/// `n` argument of our function
	n: u8,

	/// Path to cache directory, no file system caching used if not present
	#[structopt(parse(from_os_str))]
	path: Option<std::path::PathBuf>,

	/// Precalculates cache only for next value
	/// 
	/// Intermediate results are calculated faster, but more memory is used to
	/// optimize cache segments after incremental additions. 
	#[structopt(long, short)]
	sequential: bool,

	#[structopt(flatten)]
	verbose: clap_verbosity_flag::Verbosity,
}

fn print_result(n: usize, res: tree_cache::U256, start_time: Instant, start_mem: usize) {
	println!("{} {} {} {} {}",
		n,
		res,
		start_time.elapsed().as_secs_f32(),
		ALLOCATED.load(SeqCst) - start_mem,
		ALLOCATED_MAX.load(SeqCst) - start_mem
	);
}

fn main() -> Result<(), exitfailure::ExitFailure> {
	use tree_cache::Cache;

	let args = Cli::from_args();
	args.verbose.setup_env_logger("tree_cache")?;

	let start_time = Instant::now();
	let start_mem = ALLOCATED.load(SeqCst);

	let mut cache = args.path.map_or_else(|| Ok(Cache::default()), Cache::new)?;

	println!("n f(n) time[s] mem[B] max_mem[B]");
	if args.sequential {
		for n in 0..=args.n {
			let res = cache.f(n)?;
			print_result(n as usize, res, start_time, start_mem);
		}
	} else {
		for (n, r) in cache.finite_iter(args.n).enumerate() {
			print_result(n, r?, start_time, start_mem);
		}
	}

	Ok(())
}
