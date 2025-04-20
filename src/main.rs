mod fraction;

fn main() {
    // use fraction::Fraction;
    // use rand;

    // fn sqrt(n: Fraction) -> Option<Fraction> {
    //     let mut prev;
    //     let mut curr;

    //     if n.is_negative() {
    //         return None;
    //     } else if n.is_zero() {
    //         return Some(Fraction::ZERO);
    //     } else if (n - 1).is_positive() {
    //         prev = (n + 1) / 2;
    //     } else {
    //         prev = Fraction::from(1);
    //     }

    //     curr = (n / prev + prev) / 2;
    //     for _ in 0..20 {
    //         if curr - prev == Fraction::ZERO {
    //             // println!("{} {}", curr, prev);
    //             return Some(curr);
    //         }
    //         prev = curr;
    //         curr = (n / prev + prev) / 2;
    //     }
    //     Some(curr)
    // }
    
    // let range = 1e-16;
    // for _ in 0..5000 {
    //     let m = rand::random_range(0..=i64::MAX);
    //     let n = rand::random_range(1..=i64::MAX);
    //     let sqrt = sqrt(Fraction::new(m, n)).unwrap();
    //     println!("{}", (f64::from(sqrt) - (m as f64 / n as f64).sqrt()).abs());
    // }
    
    // let n = Fraction::new(i64::MAX, 4); // (9223372036854775807, 4)
    // let prev = (n + 1) / 2; // (9223372036854775811, 8)
    // let curr = (n / prev + prev) / 2; 
    // println!("{}", curr);
}

#[cfg(test)]
mod tests {
    use crate::fraction::{Fraction, ConversionError};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use rand;

    pub fn sqrt_fraction(n: Fraction) -> Option<Fraction> {
        let mut prev;
        let mut curr;
        let zero = Fraction::from(0);
        if n.is_negative() {
            return None;
        } else if n.is_zero() {
            return Some(Fraction::from(0));
        } else if (n - 1).is_positive() {
            prev = (n + 1) / 2;
        } else {
            prev = Fraction::from(1);
        }

        curr = (n / prev + prev) / 2;
        loop {
            if curr - prev == zero {
                return Some(curr);
            }
            prev = curr;
            curr = (n / prev + prev) / 2;
        }
    }

    #[test]
    fn test_creation_and_reduction() {
        let f = Fraction::new(4, 6);
        assert_eq!(f, Fraction::new(2, 3));

        let f = Fraction::new(-3, 6);
        assert_eq!(f, Fraction::new(-1, 2));

        let f = Fraction::new(3, -6);
        assert_eq!(f, Fraction::new(-1, 2));

        let f = Fraction::new(0, 5);
        assert_eq!(f, Fraction::new(0, 1));
    }

    #[test]
    fn test_arithmetic_operations() {
        let a = Fraction::new(1, 2);
        let b = Fraction::new(1, 3);
        assert_eq!(a + b, Fraction::new(5, 6));

        let a = Fraction::new(3, 4);
        let b = Fraction::new(1, 4);
        assert_eq!(a - b, Fraction::new(1, 2));

        let a = Fraction::new(2, 3);
        let b = Fraction::new(3, 4);
        assert_eq!(a * b, Fraction::new(1, 2));

        let a = Fraction::new(1, 2);
        let b = Fraction::new(2, 1);
        assert_eq!(a / b, Fraction::new(1, 4));
    }

    #[test]
    fn test_comparisons() {
        let a = Fraction::new(2, 4);
        let b = Fraction::new(1, 2);
        assert_eq!(a, b);

        let a = Fraction::new(1, 2);
        let b = Fraction::new(3, 4);
        assert!(a < b);
    }

    #[test]
    fn test_special_cases() {
        let zero = Fraction::new(0, 1);
        let a = Fraction::new(3, 4);
        assert_eq!(a + zero, a);
        assert_eq!(a - zero, a);

        assert!(Fraction::new(1, 0).is_infinity());
        assert!(Fraction::new(-1, 0).is_neg_infinity());
        assert!(Fraction::new(0, 0).is_nan());
    }

    #[test]
    fn test_assignment_operations() {
        let mut a = Fraction::new(1, 3);
        a += Fraction::new(1, 6);
        assert_eq!(a, Fraction::new(1, 2));

        let mut b = Fraction::new(3, 4);
        b -= Fraction::new(1, 4);
        assert_eq!(b, Fraction::new(1, 2));

        let mut c = Fraction::new(-7, 6);
        c *= Fraction::new(-8, 7);
        assert_eq!(c, Fraction::new(4, 3));

        let mut d = Fraction::new(-7, 6);
        d /= Fraction::new(-7, 8);
        assert_eq!(d, Fraction::new(4, 3));
    }

    #[test]
    fn test_display_formatting() {
        assert_eq!(format!("{}", Fraction::new(5, 1)), "5");
        
        assert_eq!(format!("{}", Fraction::new(3, 4)), "3/4");
        
        assert_eq!(format!("{}", Fraction::new(-2, 3)), "-2/3");
    }

    #[test]
    fn test_hash_consistency() {
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        Fraction::new(2, 4).hash(&mut hasher1);
        Fraction::new(1, 2).hash(&mut hasher2);
        
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_edge_cases() {
        let f = Fraction::new(i64::MAX, i64::MAX);
        assert_eq!(f, Fraction::new(1, 1));

        let f = Fraction::new(i64::MIN, i64::MIN);
        assert_eq!(f, Fraction::new(1, 1));
    }

    #[test]
    fn test_sign_handling() {
        assert!(Fraction::new(3, 4).is_positive());
        assert!(Fraction::new(-3, 4).is_negative());
        assert!(Fraction::new(0, 1).is_zero());
    }

    #[test]
    fn test_absolute_value() {
        let f = Fraction::new(-3, 4).abs();
        assert_eq!(f, Fraction::new(3, 4));
    }

    #[test]
    fn test_sqrt() {
        let range = 1e-10;
        for _ in 0..5000 {
            let m = rand::random_range(0..=i64::MAX);
            let n = rand::random_range(1..=i64::MAX);
            let sqrt = sqrt_fraction(Fraction::new(m, n)).unwrap();
            assert!((f64::from(sqrt) - (m as f64 / n as f64).sqrt()).abs() <= range);
        }
    }

    #[test]
    fn test_document_in_readme() {
        // safe
        let f64_val: f64 = Fraction::new(-5, 4).into();
        assert_eq!(f64_val, -1.25);
        let f64_inf: f64 = f64::from(Fraction::INFINITY);
        assert_eq!(f64_inf, f64::INFINITY);
        // unsafe
        let res_err: Result<i64, _> = Fraction::INFINITY.try_into();
        assert_eq!(res_err, Err(ConversionError::InfiniteConversion));
        let res_ok: Result<i64, _> = i64::try_from(Fraction::new(-3, 2));
        assert_eq!(res_ok, Ok(-1));

        // shrink
        let n = Fraction::new(i64::MAX, 4); // (9223372036854775807, 4)
        let prev = (n + 1) / 2;
        let curr = (n / prev + prev) / 2; 
        
        // shrink to:  3458764513820540935                    / 6                     (= 5.764607523034235e+17)
        // instead of: 85070591730234616068757836668747120633 / 147573952589676412976 (= 5.764607523034235e+17)
        assert_eq!(curr, Fraction::new(3458764513820540935, 6));
        
        // sqrt
        fn sqrt(n: Fraction) -> Option<Fraction> {
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

        assert_eq!(sqrt(Fraction::from(100)).unwrap(), Fraction::from(10));

        let a = Fraction::new(2430681237972187225, 389220499125149996);   // 6.244997998398398
        let b = Fraction::new(4909995561725655699, 786228524490300992); // 6.244997998398398
        assert!(a != b);
        assert!(a - b == Fraction::ZERO);
    }
}