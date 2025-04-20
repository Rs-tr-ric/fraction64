// A high-precision fraction lib implemented in rust.
// Copyright (C) 2025 Richard Sun
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


//! # fraction
//!
//! 提供高精度分数运算，支持特殊值处理和自动化简
//! 
//! # 核心功能
//! - 基本四则运算 (`+`, `-`, `*`, `/` 等)
//! - 与原生类型的[安全转换](crate::conversion)
//! - 特殊值处理 (INFINITY, NAN 等)
//! - 在结果溢出时候使用 shrink 将结果转化为范围内的最接近结果的最简分数

pub(crate) mod utils;

use std::{
    cmp::Ordering, fmt::{self, Display, Formatter}, hash::{Hash, Hasher}, ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign 
    }
};

use utils::U256;

/// 分数类型，使用 `i64` 存储分子分母
///
/// 维护最简分数形式，支持与基本数值类型的互操作
/// 
/// # 示例 - 基础使用
/// ```
/// use fraction::Fraction;
///
/// let a = Fraction::new(3, 4); // 3/4
/// let b = Fraction::from(2);    // 2/1
/// assert_eq!(a + b, Fraction::new(11, 4));
/// ```
///
/// # 特殊值处理
/// ```
/// # use fraction::{Fraction, ConversionError};
/// let inf = Fraction::INFINITY;
/// let nan = Fraction::NAN;
///
/// assert!(inf > Fraction::from(1000));
/// assert!(nan != nan); // NaN 不满足自反性
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Type {
    Normal,
    Infinity,
    NegInfinity,
    Zero,
    NaN
}

#[derive(Debug, PartialEq)]
pub enum ConversionError {
    OutOfRangeError, 
    NaNConversion, 
    InfiniteConversion, 
}

#[derive(Debug, Clone, Copy, Eq)]
pub struct Fraction {
    nume: i64,
    deno: i64, 
    frac_type: Type, 
}

impl Fraction {
    pub const INFINITY: Self = Self { nume: i64::MAX, deno: 1, frac_type: Type::Infinity };
    pub const NEG_INFINITY: Self = Self { nume: i64::MIN, deno: 1, frac_type: Type::NegInfinity };
    pub const NAN: Self = Self { nume: 0, deno: 0, frac_type: Type::NaN };
    pub const ZERO: Self = Self { nume: 0, deno: 1, frac_type: Type::Zero };

    pub const MAX: Self = Self { nume: i64::MAX - 1, deno: 1, frac_type: Type::Normal};
    pub const MIN: Self = Self { nume: i64::MIN + 1, deno: 1, frac_type: Type::Normal};
    pub const MIN_POSITIVE: Self = Self { nume: 1, deno: i64::MAX, frac_type: Type::Normal};

    const LIMITER: u128 = i64::MAX as u128;

    /// 创建新分数，自动化简为最简形式
    ///
    /// # 参数
    /// - `numerator`: 分子  
    /// - `denominator`: 分母 (非零)
    ///
    /// # Panics
    /// 当分母为零时触发 panic
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let normal = Fraction::new(4, -2); // -2/1
    /// assert_eq!(normal, Fraction::from(-2));
    /// 
    /// let inf = Fraction::new(1, 0); // inf
    /// assert_eq!(inf, Fraction::INFINITY);
    /// ```
    pub fn new(nume: i64, deno: i64) -> Self {
        let frac_type = Self::determine_frac_type(nume, deno);
        match frac_type {
            Type::Infinity => Self::INFINITY, 
            Type::NegInfinity => Self::NEG_INFINITY, 
            Type::NaN => Self::NAN, 
            Type::Zero => Self::ZERO, 
            Type::Normal => {
                let sign = nume.signum() * deno.signum();
        
                let (nume, deno) = (nume.unsigned_abs() as u128, deno.unsigned_abs() as u128);
                let gcd_val = Self::gcd(nume, deno);
                let (nume, deno) = Self::shrink(nume / gcd_val, deno / gcd_val);
                let (nume, deno) = (nume as i64 * sign, deno as i64);
        
                Self {
                    nume, 
                    deno, 
                    frac_type: Self::determine_frac_type(nume, deno)
                }
            }
        }
    }

    fn determine_frac_type(nume: i64, deno: i64) -> Type {
        if deno == 0 { 
            match nume.signum() {
                1 => Type::Infinity,
                -1 => Type::NegInfinity,
                _ => Type::NaN,
            }
        } else if nume == 0 {
            Type::Zero
        } else if deno == 1 {
            match nume {
                i64::MAX => Type::Infinity,
                i64::MIN => Type::NegInfinity,
                _ => Type::Normal,
            }
        } else {
            Type::Normal
        }
    }

    fn gcd<T>(a: T, b: T) -> T
    where
        T: Rem<Output = T> + From<u8> + Eq + Copy // + std::fmt::Display
    {
        let (mut a, mut b) = (a, b);
        while b != T::from(0u8) {
            (a, b) = (b, a % b);
        };
        a
    }

    fn lcm<T>(a: T, b: T) -> (T, T, T)
    where
        T: Div<Output = T> + Rem<Output = T> + From<u8> + Eq + Copy // + std::fmt::Display
    {
        let gcd = Self::gcd(a, b);
        (b / gcd, a / gcd, gcd)
    }

    fn shrink(nume: u128, deno: u128) -> (u64, u64) {
        let nume_abs = nume;
        let deno_abs = deno;
            
        if nume_abs <= Self::LIMITER && deno_abs <= Self::LIMITER {
            return (nume as u64, deno as u64);
        }

        let (mut p_0, mut q_0, mut p_1, mut q_1) = (0, 1, 1, 0); // [0, +inf)
        let (mut nume, mut deno) = (nume_abs, deno_abs);
        loop {
            let q = nume / deno;
            let p_2 = p_0 + q * p_1;
            let q_2 = q_0 + q * q_1;
            
            if p_2 > Self::LIMITER || q_2 > Self::LIMITER {
                break;
            }

            (p_0, q_0, p_1, q_1) = (p_1, q_1, p_2, q_2);
            (nume, deno) = (deno, nume - q * deno);
        }
        let (k_q, k_p) = {
            let k_q = if q_1 != 0 {
                (Self::LIMITER - q_0) / q_1
            } else {
                return (i64::MAX as u64, 1); // q_1 == 0 <=> inf
            };
        
            let k_p = if p_1 != 0 {
                (Self::LIMITER - p_0) / p_1
            } else {
                return (0, 1); // p_1 == 0 <=> 0
            };
        
            (k_q, k_p)
        };
        let k = k_q.min(k_p).max(0);

        let (nume_1, deno_1) = (p_1, q_1);
        let (nume_2, deno_2) = (p_0 + k * p_1, q_0 + k * q_1);
        
        let (deno_abs, nume_abs) = (U256::from(deno_abs), U256::from(nume_abs));
        let (nume_1, deno_1) = (U256::from(nume_1), U256::from(deno_1));
        let (nume_2, deno_2) = (U256::from(nume_2), U256::from(deno_2));

        let d_1 = (nume_1 * deno_abs).sub_abs(nume_abs * deno_1);
        let d_2 = (nume_2 * deno_abs).sub_abs(nume_abs * deno_2);

        if d_1 * deno_2 <= d_2 * deno_1 { (nume_1.into(), deno_1.into()) } else { (nume_2.into(), deno_2.into()) }
    }

    /// 获取符号
    ///
    /// # 返回值
    /// `Self`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert_eq!(a1.sign(), Fraction::from(1));
    /// let a2 = Fraction::new(-2, 3);
    /// assert_eq!(a2.sign(), Fraction::from(-1));
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(zero.sign().is_zero());
    /// let nan = Fraction::NAN;
    /// assert!(nan.sign().is_nan());
    /// let inf = Fraction::INFINITY;
    /// assert_eq!(inf.sign(), Fraction::from(1));
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert_eq!(neg_inf.sign(), Fraction::from(-1));
    /// ```
    pub fn sign(&self) -> Self {
        match self.frac_type {
            Type::NaN => Self::NAN,
            Type::Zero => Self::ZERO,
            Type::Infinity => Self { 
                nume: 1, 
                deno: 1, 
                frac_type: Type::Normal
            },
            Type::NegInfinity => Self { 
                nume: -1, 
                deno: 1, 
                frac_type: Type::Normal
            },
            Type::Normal => if self.nume >= 0 { Self { 
                nume: 1, 
                deno: 1, 
                frac_type: Type::Normal
            } } else { Self { 
                nume: -1, 
                deno: 1, 
                frac_type: Type::Normal
            } }
        }
    }
    
    fn i64_sign(&self) -> i64 {
        match self.frac_type {
            Type::NaN => 0,
            Type::Zero => 0,
            Type::Infinity => 1,
            Type::NegInfinity => -1,
            Type::Normal => if self.nume > 0 { 1 } else { -1 }
        }
    }

    /// 正值返回 true，否则返回 false
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert!(a1.is_positive());
    /// let a2 = Fraction::new(-2, 3);
    /// assert!(!a2.is_positive());
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(!zero.is_positive());
    /// let nan = Fraction::NAN;
    /// assert!(!nan.is_positive());
    /// let inf = Fraction::INFINITY;
    /// assert!(inf.is_positive());
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(!neg_inf.is_positive());
    /// ```
    pub fn is_positive(&self) -> bool {
        match self.frac_type {
            Type::NegInfinity | Type::NaN | Type::Zero => false, 
            Type::Infinity => true,
            _ => self.nume > 0
        }
    }

    /// 负值返回 true，否则返回 false
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert!(!a1.is_negative());
    /// let a2 = Fraction::new(-2, 3);
    /// assert!(a2.is_negative());
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(!zero.is_negative());
    /// let nan = Fraction::NAN;
    /// assert!(!nan.is_negative());
    /// let inf = Fraction::INFINITY;
    /// assert!(!inf.is_negative());
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(neg_inf.is_negative());
    /// ```
    pub fn is_negative(&self) -> bool {
        match self.frac_type {
            Type::Infinity | Type::NaN | Type::Zero => false, 
            Type::NegInfinity => true,
            _ => self.nume < 0
        }
    }

    /// 零值返回 true，否则返回 false
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert!(!a1.is_zero());
    /// let a2 = Fraction::new(-2, 3);
    /// assert!(!a2.is_zero());
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(zero.is_zero());
    /// let nan = Fraction::NAN;
    /// assert!(!nan.is_zero());
    /// let inf = Fraction::INFINITY;
    /// assert!(!inf.is_zero());
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(!neg_inf.is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.frac_type == Type::Zero
    }

    /// 正无穷返回 true，否则返回 false
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert!(!a1.is_infinity());
    /// let a2 = Fraction::new(-2, 3);
    /// assert!(!a2.is_infinity());
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(!zero.is_infinity());
    /// let nan = Fraction::NAN;
    /// assert!(!nan.is_infinity());
    /// let inf = Fraction::INFINITY;
    /// assert!(inf.is_infinity());
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(!neg_inf.is_infinity());
    /// ```
    pub fn is_infinity(&self) -> bool {
        self.frac_type == Type::Infinity
    }

    /// 负无穷返回 true，否则返回 false
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert!(!a1.is_neg_infinity());
    /// let a2 = Fraction::new(-2, 3);
    /// assert!(!a2.is_neg_infinity());
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(!zero.is_neg_infinity());
    /// let nan = Fraction::NAN;
    /// assert!(!nan.is_neg_infinity());
    /// let inf = Fraction::INFINITY;
    /// assert!(!inf.is_neg_infinity());
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(neg_inf.is_neg_infinity());
    /// ```
    pub fn is_neg_infinity(&self) -> bool {
        self.frac_type == Type::NegInfinity
    }

    /// NaN 值返回 true，否则返回 false
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert!(!a1.is_nan());
    /// let a2 = Fraction::new(-2, 3);
    /// assert!(!a2.is_nan());
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(!zero.is_nan());
    /// let nan = Fraction::NAN;
    /// assert!(nan.is_nan());
    /// let inf = Fraction::INFINITY;
    /// assert!(!inf.is_nan());
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(!neg_inf.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.frac_type == Type::NaN
    }

    /// 非特数值（不包括零值）返回 true，否则返回 false
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// assert!(a1.is_normal());
    /// let a2 = Fraction::new(-2, 3);
    /// assert!(a2.is_normal());
    ///
    /// let zero = Fraction::ZERO;
    /// assert!(!zero.is_normal());
    /// let nan = Fraction::NAN;
    /// assert!(!nan.is_normal());
    /// let inf = Fraction::INFINITY;
    /// assert!(!inf.is_normal());
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(!neg_inf.is_normal());
    /// ```
    pub fn is_normal(&self) -> bool {
        self.frac_type == Type::Normal
    }

    /// 获取绝对值，保持特殊值语义
    ///
    /// # 返回值
    /// `Self`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a = Fraction::new(-2, 3);
    /// assert_eq!(a.abs(), Fraction::new(2, 3));
    ///
    /// let inf = Fraction::INFINITY;
    /// assert_eq!(inf.abs(), Fraction::INFINITY);
    /// ```
    pub fn abs(&self) -> Self {
        match self.frac_type {
            Type::NegInfinity | Type::Infinity => Self::INFINITY, 
            Type::NaN => Self::NAN, 
            Type::Zero => Self::ZERO, 
            _ => Self {
                nume: self.nume.abs(), 
                deno: self.deno, 
                frac_type: self.frac_type
            }
        }
    }

    /// 获取倒数，保持特殊值语义
    ///
    /// # 返回值
    /// `Self`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a = Fraction::new(2, 3);
    /// assert_eq!(a.reciprocal(), Fraction::new(3, 2));
    ///
    /// let inf = Fraction::INFINITY;
    /// assert!(inf.reciprocal().is_zero());
    /// ```
    pub fn reciprocal(&self) -> Self {
        match self.frac_type {
            Type::Infinity => Self::ZERO, 
            Type::NegInfinity => Self::ZERO, 
            Type::NaN => Self::NAN, 
            Type::Zero => Self::INFINITY, 
            _ => Self { 
                nume: self.deno.abs() * self.i64_sign(), 
                deno: self.nume.abs(), 
                frac_type: self.frac_type
            }
        }
    }

    // operations
    fn get_add_type(self, rhs: Self) -> Type {
        match (self.frac_type, rhs.frac_type) {
            // NaN
            (Type::NaN, _) | (_, Type::NaN) => Type::NaN,
            
            // (+inf / -inf) + (+inf / -inf)
            (Type::Infinity, Type::NegInfinity) | (Type::NegInfinity, Type::Infinity) => Type::NaN,
            (Type::Infinity, _) | (_, Type::Infinity) => Type::Infinity,
            (Type::NegInfinity, _) | (_, Type::NegInfinity) => Type::NegInfinity,
            
            // 0 + 0
            (Type::Zero, Type::Zero) => Type::Zero,
            
            // normal + normal
            _ => Type::Normal, 
        }
    }

    fn normal_add(self, rhs: Self) -> (i64, i64) {
        let (a, b) = (self.nume as i128, self.deno as i128);
        let (c, d) = (rhs.nume as i128, rhs.deno as i128);

        let (e, f, gcd_bd) = Self::lcm(b, d);
        let (nume, deno) = (
            a * e + c * f, e * f * gcd_bd
        );

        let sign = nume.signum() as i64;
        let (u_num, u_den) = (nume.unsigned_abs(), deno as u128);

        let gcd = Self::gcd(u_num, u_den);
        let (simplified_num, simplified_den) = (u_num / gcd, u_den / gcd);

        let (num, den) = Self::shrink(simplified_num, simplified_den);

        (num as i64 * sign, den as i64)
    }

    fn get_mul_type(self, rhs: Self) -> Type {
        match (self.frac_type, rhs.frac_type) {
            // NaN
            (Type::NaN, _) | (_, Type::NaN) => Type::NaN,
        
            // inf * zero
            (Type::Infinity | Type::NegInfinity, Type::Zero) |
            (Type::Zero, Type::Infinity | Type::NegInfinity) => Type::NaN,
        
            // inf * inf | -inf * -inf
            (Type::Infinity, Type::Infinity) | (Type::NegInfinity, Type::NegInfinity) => Type::Infinity,
        
            // inf * -inf | -inf * inf
            (Type::Infinity, Type::NegInfinity) | (Type::NegInfinity, Type::Infinity) => Type::NegInfinity,
        
            // 0 * 0/normal | 0/normal * 0
            (Type::Zero, _) | (_, Type::Zero) => Type::Zero,
        
            // normal * inf / normal * inf 
            (Type::Normal, Type::Infinity) | (Type::Infinity, Type::Normal) => 
                if self.is_negative() ^ rhs.is_negative() { Type::NegInfinity } else { Type::Infinity },
            (Type::Normal, Type::NegInfinity) | (Type::NegInfinity, Type::Normal) => 
                if self.is_negative() ^ rhs.is_negative() { Type::NegInfinity } else { Type::Infinity },

            // normal * normal
            (Type::Normal, Type::Normal) => Type::Normal
        }

    }

    fn normal_mul(self, rhs: Self) -> (i64, i64) {
        let (a, b) = (self.nume.unsigned_abs() as u128, self.deno.unsigned_abs() as u128);
        let (c, d) = (rhs.nume.unsigned_abs() as u128, rhs.deno.unsigned_abs() as u128);

        let gcd_ad = Self::gcd(a, d);
        let gcd_bc = Self::gcd(b, c);
        let a = a / gcd_ad;
        let d = d / gcd_ad;
        let b = b / gcd_bc;
        let c = c / gcd_bc;

        let (nume, deno) = (a * c, b * d);
        // println!("mul_impl {} {}", nume, deno);
        let (nume, deno) = Self::shrink(nume, deno);
        
        (nume as i64 * self.i64_sign() * rhs.i64_sign(), deno as i64)
    }
}

impl<T: Into<Fraction>> Add<T> for Fraction {
    type Output = Self;

    /// 分数加法，自动处理特殊值
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// let a = Fraction::new(1, 2);
    /// let b = Fraction::new(1, 3);
    /// assert_eq!(a + b, Fraction::new(5, 6));
    /// assert_eq!(a + 1, Fraction::new(3, 2));
    /// 
    /// let inf = Fraction::INFINITY;
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// let nan = Fraction::NAN; 
    /// let zero = Fraction::ZERO;
    /// assert!((inf + neg_inf).is_nan());
    /// assert!((inf + inf).is_infinity());
    /// assert!((a + neg_inf).is_neg_infinity());
    /// assert!((a + nan).is_nan());
    /// ```
    fn add(self, rhs: T) -> Self::Output {
        let rhs: Self = rhs.into();
        let add_type = self.get_add_type(rhs);
        match add_type {
            Type::Infinity => Self::INFINITY, 
            Type::NegInfinity => Self::NEG_INFINITY, 
            Type::NaN => Self::NAN, 
            Type::Zero => Self::ZERO, 
            Type::Normal => {
                let (nume, deno) = self.normal_add(rhs);
                Self { 
                    nume, 
                    deno, 
                    frac_type: Self::determine_frac_type(nume, deno)
                }
            }
        }
    }
}

impl<T: Into<Fraction>> Sub<T> for Fraction {
    type Output = Self;

    /// 分数减法，自动处理特殊值
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// let a = Fraction::new(1, 2);
    /// let b = Fraction::new(1, 3);
    /// assert_eq!(a - b, Fraction::new(1, 6));
    /// assert_eq!(a - 1, Fraction::new(-1, 2));
    /// 
    /// let inf = Fraction::INFINITY;
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// let nan = Fraction::NAN; 
    /// let zero = Fraction::ZERO;
    /// assert!((inf - inf).is_nan());
    /// assert!((inf - neg_inf).is_infinity());
    /// assert!((a - inf).is_neg_infinity());
    /// assert!((a - nan).is_nan());
    /// ```
    fn sub(self, rhs: T) -> Self::Output {
        let rhs: Self = -rhs.into();
        self + rhs
    }
}

impl<T: Into<Fraction>> Mul<T> for Fraction {
    type Output = Self;

    /// 分数乘法，自动处理特殊值
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// let a = Fraction::new(1, 2);
    /// let b = Fraction::new(1, 3);
    /// assert_eq!(a * b, Fraction::new(1, 6));
    /// assert_eq!(a * 3, Fraction::new(3, 2));
    /// 
    /// let inf = Fraction::INFINITY;
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// let nan = Fraction::NAN; 
    /// let zero = Fraction::ZERO;
    /// assert!((inf * neg_inf).is_neg_infinity());
    /// assert!((inf * zero).is_nan());
    /// assert!((a * neg_inf).is_neg_infinity());
    /// assert!((a * nan).is_nan());
    /// ```
    fn mul(self, rhs: T) -> Self::Output {
        let rhs: Self = rhs.into();
        let add_type = self.get_mul_type(rhs);
        match add_type {
            Type::Infinity => Self::INFINITY, 
            Type::NegInfinity => Self::NEG_INFINITY, 
            Type::NaN => Self::NAN, 
            Type::Zero => Self::ZERO, 
            Type::Normal => {
                let (nume, deno) = self.normal_mul(rhs);
                Self { 
                    nume, 
                    deno, 
                    frac_type: Self::determine_frac_type(nume, deno)
                }
            }
        }
    }
}

impl<T: Into<Fraction>> Div<T> for Fraction {
    type Output = Self;

    /// 分数除法，自动处理特殊值
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// let a = Fraction::new(1, 2);
    /// let b = Fraction::new(1, 3);
    /// assert_eq!(a / b, Fraction::new(3, 2));
    /// assert_eq!(a / 3, Fraction::new(1, 6));
    /// 
    /// let inf = Fraction::INFINITY;
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// let nan = Fraction::NAN; 
    /// let zero = Fraction::ZERO;
    /// assert!((inf / neg_inf).is_nan());
    /// assert!((inf / zero).is_infinity());
    /// assert!((a / neg_inf).is_zero());
    /// assert!((a / nan).is_nan());
    /// ```
    fn div(self, rhs: T) -> Self::Output {
        let rhs: Self = rhs.into().reciprocal();
        self * rhs
    }
}

impl<T: Into<Fraction>> AddAssign<T> for Fraction {
    /// 实现 `+=` 操作
    /// 
    /// 在数值上与 `+` 的行为相同
    /// 
    /// # 示例
    /// ```rust
    /// # use fraction::Fraction;
    /// let mut a = Fraction::new(1, 2);
    /// let mut b = Fraction::new(1, 3);
    /// a += b;
    /// b += 3;
    /// assert_eq!(a, Fraction::new(5, 6));
    /// assert_eq!(b, Fraction::new(10, 3));
    /// ```
    fn add_assign(&mut self, rhs: T) {
        let rhs: Self = rhs.into();
        let add_type = self.get_add_type(rhs);
        match add_type {
            Type::Infinity => *self = Self::INFINITY, 
            Type::NegInfinity => *self = Self::NEG_INFINITY, 
            Type::NaN => *self = Self::NAN, 
            Type::Zero => *self = Self::ZERO, 
            Type::Normal => {
                (self.nume, self.deno) = self.normal_add(rhs);
                self.frac_type = Self::determine_frac_type(self.nume, self.deno);
            }
        }
    }
}

impl<T: Into<Fraction>> SubAssign<T> for Fraction {
    /// 实现 `-=` 操作
    /// 
    /// 在数值上与 `-` 的行为相同
    /// 
    /// # 示例
    /// ```rust
    /// # use fraction::Fraction;
    /// let mut a = Fraction::new(1, 2);
    /// let mut b = Fraction::new(1, 3);
    /// a -= b;
    /// b -= 3;
    /// assert_eq!(a, Fraction::new(1, 6));
    /// assert_eq!(b, Fraction::new(-8, 3));
    /// ```
    fn sub_assign(&mut self, rhs: T) {
        let rhs: Self = rhs.into();
        *self += -rhs;
    }
}

impl<T: Into<Fraction>> MulAssign<T> for Fraction {
    /// 实现 `*=` 操作
    /// 
    /// 在数值上与 `*` 的行为相同
    /// 
    /// # 示例
    /// ```rust
    /// # use fraction::Fraction;
    /// let mut a = Fraction::new(1, 2);
    /// let mut b = Fraction::new(1, 3);
    /// a *= b;
    /// b *= 3;
    /// assert_eq!(a, Fraction::new(1, 6));
    /// assert_eq!(b, Fraction::from(1));
    /// ```
    fn mul_assign(&mut self, rhs: T) {
        let rhs: Self = rhs.into();
        let add_type = self.get_add_type(rhs);
        match add_type {
            Type::Infinity => *self = Self::INFINITY, 
            Type::NegInfinity => *self = Self::NEG_INFINITY, 
            Type::NaN => *self = Self::NAN, 
            Type::Zero => *self = Self::ZERO, 
            Type::Normal => {
                (self.nume, self.deno) = self.normal_mul(rhs);
                self.frac_type = Self::determine_frac_type(self.nume, self.deno);
            }
        }
    }
}

impl<T: Into<Fraction>> DivAssign<T> for Fraction {
    /// 实现 `/=` 操作
    /// 
    /// 在数值上与 `/` 的行为相同
    /// 
    /// # 示例
    /// ```rust
    /// # use fraction::Fraction;
    /// let mut a = Fraction::new(1, 2);
    /// let mut b = Fraction::new(1, 3);
    /// a /= b;
    /// b /= 3;
    /// assert_eq!(a, Fraction::new(3, 2));
    /// assert_eq!(b, Fraction::new(1, 9));
    /// ```
    fn div_assign(&mut self, rhs: T) {
        let rhs: Self = rhs.into();
        *self *= rhs.reciprocal();
    }
}

impl Neg for Fraction {
    type Output = Self;

    /// 取反，自动处理特殊值
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// let a = Fraction::new(1, 2);
    /// assert_eq!(-a, Fraction::new(-1, 2));
    /// 
    /// let inf = Fraction::INFINITY;
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// let nan = Fraction::NAN; 
    /// let zero = Fraction::ZERO;
    /// assert!((-nan).is_nan());
    /// assert!((-inf).is_neg_infinity());
    /// assert!((-neg_inf).is_infinity());
    /// assert!((-zero).is_zero());
    /// ```
    fn neg(self) -> Self::Output {
        match self.frac_type {
            Type::Infinity => Self::NEG_INFINITY, 
            Type::NegInfinity => Self::INFINITY, 
            Type::NaN => Self::NAN, 
            Type::Zero => Self::ZERO, 
            Type::Normal => {
                Self {
                    nume: -self.nume, 
                    deno: self.deno, 
                    frac_type: Type::Normal
                }
            }
        }
    }
}

impl Display for Fraction {
    /// 格式化输出
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// let a = Fraction::new(1, 2);
    /// let b = Fraction::new(1, -2);
    /// assert_eq!(a.to_string(), "1/2");
    /// assert_eq!(b.to_string(), "-1/2");
    /// 
    /// let inf = Fraction::INFINITY;
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// let nan = Fraction::NAN; 
    /// let zero = Fraction::ZERO;
    /// assert_eq!(inf.to_string(), "inf");
    /// assert_eq!(neg_inf.to_string(), "-inf");
    /// assert_eq!(nan.to_string(), "nan");
    /// assert_eq!(zero.to_string(), "0");
    /// ```
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self.frac_type {
            Type::Infinity => write!(f, "inf"), 
            Type::NegInfinity => write!(f, "-inf"), 
            Type::NaN => write!(f, "nan"), 
            Type::Zero => write!(f, "0"), 
            Type::Normal => if self.deno == 1 {
                write!(f, "{}", self.nume)
            } else {
                write!(f, "{}/{}", self.nume, self.deno)
            }
        }
    }
}

impl PartialEq for Fraction {
    /// 判断是否相等
    /// 
    /// 比较逻辑处理以下特殊值：
    /// - 无穷大（`INFINITY`/`NEG_INFINITY`）
    /// - NaN
    /// - 零
    /// - 普通分数
    ///
    /// # 取等规则
    /// 1. **NaN 参与比较**：任意操作数为 NaN 时返回 `false`
    /// 2. **特殊值比较**：非 NaN 的特殊值仅与自身下相等
    /// 3. **普通分数比较**：直接判断分数的分子分母是否全部相等
    ///
    /// # 返回值
    /// `bool`
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// 
    /// let a1 = Fraction::new(2, 3);
    /// let a2 = Fraction::new(-4, -6);
    /// assert!(a1 == a2);
    ///
    /// let zero = Fraction::ZERO;
    /// let nan = Fraction::NAN;
    /// let inf = Fraction::INFINITY;
    /// let neg_inf = Fraction::NEG_INFINITY;
    /// assert!(zero != a1);
    /// assert!(nan != nan);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        match (self.frac_type, other.frac_type) {
            (Type::NaN, _) | (_, Type::NaN) => false, 
            (_, _) => self.frac_type == self.frac_type && self.nume == other.nume && self.deno == other.deno
        }
    }
}

impl PartialOrd for Fraction {
    /// 实现分数的比较运算，遵循扩展实数系统的规则
    ///
    /// 比较逻辑处理以下特殊值：
    /// - 无穷大（`INFINITY`/`NEG_INFINITY`）
    /// - NaN
    /// - 零
    /// - 普通分数
    ///
    /// # 比较规则
    /// 1. **NaN 参与比较**：任意操作数为 NaN 时返回 `None`
    /// 2. **无穷大比较**：
    ///    - `INFINITY` 大于所有非 NaN 值（包括自身相等性）
    ///    - `NEG_INFINITY` 小于所有非 NaN 值（包括自身相等性）
    /// 3. **普通分数比较**：使用交叉相乘算法避免精度损失
    ///
    /// # 返回值
    /// 返回 `Option<Ordering>`：
    /// - `Some(Ordering::Greater)`：当前值大于比较值
    /// - `Some(Ordering::Less)`：当前值小于比较值
    /// - `Some(Ordering::Equal)`：数学相等
    /// - `None`：存在 NaN 无法比较
    ///
    /// # 示例
    /// ```
    /// # use fraction::Fraction;
    /// # use std::cmp::Ordering;
    ///
    /// let a = Fraction::new(3, 4);
    /// let b = Fraction::new(2, 3);
    /// assert_eq!(a.partial_cmp(&b), Some(Ordering::Greater));
    /// 
    /// let inf = Fraction::INFINITY;
    /// let zero = Fraction::ZERO;
    /// let nan = Fraction::NAN;
    /// assert_eq!(inf.partial_cmp(&zero), Some(Ordering::Greater));
    /// assert_eq!(nan.partial_cmp(&a), None);
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.frac_type, other.frac_type) {
            (Type::NaN, _) | (_, Type::NaN) => None, 

            // self_type: infinity
            (Type::Infinity, Type::NegInfinity) | (Type::Infinity, Type::Normal) |
            (Type::Infinity, Type::Zero) => Some(Ordering::Greater),  
            (Type::Infinity, Type::Infinity) => Some(Ordering::Equal), 

            // self_type: neg_infinity
            (Type::NegInfinity, Type::Infinity) | (Type::NegInfinity, Type::Normal) |
            (Type::NegInfinity, Type::Zero) => Some(Ordering::Less), 
            (Type::NegInfinity, Type::NegInfinity) => Some(Ordering::Equal), 

            // self_type: normal
            (Type::Normal, Type::Infinity) => Some(Ordering::Less), 
            (Type::Normal, Type::NegInfinity) => Some(Ordering::Less), 
            (Type::Normal, _) | (Type::Zero, _) => {
                let (a, b) = (self.nume as i64, self.deno as i64);
                let (c, d) = (other.nume as i64, other.deno as i64);
                Some((a * d).cmp(&(b * c)))
            }
        }
    }
}

// panic! when NaN
// impl Ord for Fraction {
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }

macro_rules! impl_from_safe {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Fraction {
                fn from(value: $t) -> Self {
                    let value = value as i64;
                    match Self::determine_frac_type(value, 1) {
                        Type::Zero => Self::ZERO, 
                        Type::Normal => Self {
                            nume: value, 
                            deno: 1, 
                            frac_type: Type::Normal
                        }, 
                        _ => Self::NAN
                    }
                }
            }
        )*
    };
}

macro_rules! impl_from_unsigned_unsafe {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Fraction {
                fn from(value: $t) -> Self {
                    let value = value as i64;
                    match Self::determine_frac_type(value, 1) {
                        Type::Zero => Self::ZERO, 
                        Type::Infinity => Self::INFINITY, 
                        Type::Normal => Self {
                            nume: value, 
                            deno: 1, 
                            frac_type: Type::Normal
                        }, 
                        _ => Self::NAN
                    }
                }
            }
        )*
    };
}

macro_rules! impl_from_signed_unsafe {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Fraction {
                fn from(value: $t) -> Self {
                    let value = value as i64;
                    match Self::determine_frac_type(value, 1) {
                        Type::Zero => Self::ZERO, 
                        Type::Infinity => Self::INFINITY, 
                        Type::NegInfinity => Self::NEG_INFINITY, 
                        Type::Normal => Self {
                            nume: value, 
                            deno: 1, 
                            frac_type: Type::Normal
                        }, 
                        _ => Self::NAN
                    }
                }
            }
        )*
    };
}

impl_from_safe!(u8, u16, u32, i8, i16, i32, i64);
impl_from_unsigned_unsafe!(u64, u128);
impl_from_signed_unsafe!(i128);

macro_rules! impl_from_for_float {
    ($($t:ty),*) => {
        $(
            impl From<Fraction> for $t {
                fn from(value: Fraction) -> Self {
                    match value.frac_type {
                        Type::Infinity => <$t>::INFINITY, 
                        Type::NegInfinity => <$t>::NEG_INFINITY, 
                        Type::NaN => <$t>::NAN, 
                        Type::Zero => 0.0, 
                        _ => value.nume as $t / value.deno as $t
                    }
                }
            }
        )*
    };
}

impl_from_for_float!(f32, f64);

macro_rules! impl_try_from_for_integer_with_lower_capacity {
    ($($t:ty),*) => {
        $(
            impl TryFrom<Fraction> for $t {
                type Error = ConversionError;

                fn try_from(value: Fraction) -> Result<Self, Self::Error> {
                    match value.frac_type {
                        Type::Infinity | Type::NegInfinity => Err(ConversionError::InfiniteConversion), 
                        Type::NaN => Err(ConversionError::NaNConversion), 
                        Type::Zero => Ok(0), 
                        Type::Normal => {
                            let integer = value.nume / value.deno;
                            if Self::MIN as i64 <= integer && integer <= Self::MAX as i64 {
                                Ok(integer as $t)
                            } else {
                                Err(ConversionError::OutOfRangeError)
                            }
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! impl_try_from_for_unsigned_integer_with_greater_capacity {
    ($($t:ty),*) => {
        $(
            impl TryFrom<Fraction> for $t {
                type Error = ConversionError;

                fn try_from(value: Fraction) -> Result<Self, Self::Error> {
                    match value.frac_type {
                        Type::Infinity | Type::NegInfinity => Err(ConversionError::InfiniteConversion), 
                        Type::NaN => Err(ConversionError::NaNConversion), 
                        Type::Zero => Ok(0), 
                        Type::Normal => {
                            let integer = value.nume / value.deno;
                            if integer >= 0 {
                                Ok(integer as $t)
                            } else {
                                Err(ConversionError::OutOfRangeError)
                            }
                        }
                    }
                }
            }
        )*
    };
}

macro_rules! impl_try_from_for_signed_integer_with_greater_capacity {
    ($($t:ty),*) => {
        $(
            impl TryFrom<Fraction> for $t {
                type Error = ConversionError;

                fn try_from(value: Fraction) -> Result<Self, Self::Error> {
                    match value.frac_type {
                        Type::Infinity | Type::NegInfinity => Err(ConversionError::InfiniteConversion), 
                        Type::NaN => Err(ConversionError::NaNConversion), 
                        Type::Zero => Ok(0), 
                        Type::Normal => {
                            let integer = value.nume / value.deno;
                            Ok(integer as $t)
                        }
                    }
                }
            }
        )*
    };
}

impl_try_from_for_integer_with_lower_capacity!(i8, i16, i32, u8, u16, u32);
impl_try_from_for_unsigned_integer_with_greater_capacity!(u64, u128);
impl_try_from_for_signed_integer_with_greater_capacity!(i64, i128);

impl Hash for Fraction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.nume.hash(state);
        self.deno.hash(state);
    }
}