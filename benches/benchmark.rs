use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fraction::Fraction;
use rand;

fn sqrt_fraction(n: Fraction) -> Option<Fraction> {
    let mut prev;
    let mut curr;
    if n.is_negative() {
        return None;
    } else if n.is_zero() {
        return Some(Fraction::ZERO);
    } else if (n - 1).is_positive() {
        prev = (n + 1) / 2;
    } else {
        prev = Fraction::from(1);
    }

    curr = (n / prev + prev) / 2;
    loop {
        if curr - prev == Fraction::ZERO {
            return Some(curr);
        }
        prev = curr;
        curr = (n / prev + prev) / 2;
    }
}

fn sqrt_f64(n: f64) -> Option<f64> {
    let mut prev;
    let mut curr;
    if n.is_sign_negative() {
        return None;
    } else if n > 1.0 {
        prev = (n + 1.0) / 2.0;
    } else {
        prev = 1.0;
    }

    loop {
        curr = (n / prev + prev) / 2.0;
        if (curr - prev).abs() < 1e-16 {
            return Some(curr);
        }
        prev = curr;
    }
}

fn benchmark_fraction(c: &mut Criterion) {
    c.bench_function(
        "sqrt(fraction)", 
        |b| b.iter(
            || sqrt_fraction(black_box(Fraction::new(rand::random_range(0..i64::MAX), rand::random_range(1..i64::MAX))))));
}

fn benchmark_f64(c: &mut Criterion) {
    c.bench_function(
        "sqrt(f64)", 
        |b| b.iter(
            || sqrt_f64(black_box(rand::random_range(0..i64::MAX) as f64 / rand::random_range(0..i64::MAX) as f64))
        )
    );
}

criterion_group!(benches, benchmark_fraction, benchmark_f64);
criterion_main!(benches);