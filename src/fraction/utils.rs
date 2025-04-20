use std::ops::Mul;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, Eq)]
pub struct U256 {
    chunks: [u64; 4], // 低位在前（chunks[0] 是最低位）
}

impl U256 {
    // sub then abs
    pub fn sub_abs(self, rhs: Self) -> U256 {
        let (mut diff, borrow) = self.sub_with_borrow(rhs);

        if borrow {
            // 手动计算补码：!diff + 1（无需实现 Not trait）
            diff.chunks[0] = !diff.chunks[0];
            diff.chunks[1] = !diff.chunks[1];
            diff.chunks[2] = !diff.chunks[2];
            diff.chunks[3] = !diff.chunks[3];
            diff.add_one();
            diff
        } else { diff }
    }

    /// sub with borrow
    fn sub_with_borrow(&self, rhs: U256) -> (U256, bool) {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;

        for i in 0..4 {
            let rhs_val = rhs.chunks[i] + borrow;
            let (diff, b) = self.chunks[i].overflowing_sub(rhs_val);
            result[i] = diff;
            borrow = b as u64; // borrow = 0 or 1
        }

        (U256 { chunks: result }, borrow != 0)
    }

    // in-place add one
    fn add_one(&mut self) {
        let mut carry = 1u64;

        for chunk in &mut self.chunks {
            let sum = (*chunk as u128) + carry as u128;
            *chunk = sum as u64;
            carry = (sum >> 64) as u64;
            if carry == 0 {
                break;
            }
        }
    }
}

impl Mul for U256 {
    type Output = Self;

    /// implement `*`
    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = [0u64; 8]; // temp result (512b)

        for i in 0..4 {
            let a = self.chunks[i];
            let mut carry = 0u64;

            for j in 0..4 {
                let b = rhs.chunks[j];
                let product = (a as u128) * (b as u128) + carry as u128;

                // sum into specific place
                let idx = i + j;
                let sum = result[idx] as u128 + (product & 0xffff_ffff_ffff_ffff);
                result[idx] = sum as u64;

                // handle carry
                carry = (product >> 64) as u64 + (sum >> 64) as u64;
            }

            // handle carries
            if i + 4 < 8 {
                result[i + 4] += carry;
            }
        }

        // clip to U256
        U256 {
            chunks: [result[0], result[1], result[2], result[3]],
        }
    }
}

impl From<U256> for u64 {
    fn from(value: U256) -> Self {
        value.chunks[0]
    }
}

impl From<u128> for U256 {
    fn from(value: u128) -> Self {
        U256 { chunks: [value as u64, (value >> 64) as u64, 0, 0] }
    }
}

impl PartialEq for U256 {
    fn eq(&self, other: &Self) -> bool {
        self.chunks[3] == other.chunks[3]
            && self.chunks[2] == other.chunks[2]
            && self.chunks[1] == other.chunks[1]
            && self.chunks[0] == other.chunks[0]
    }
}

impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other)) // do Ord trait
    }
}

impl Ord for U256 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // hight => low
        match self.chunks[3].cmp(&other.chunks[3]) {
            Ordering::Equal => match self.chunks[2].cmp(&other.chunks[2]) {
                Ordering::Equal => match self.chunks[1].cmp(&other.chunks[1]) {
                    Ordering::Equal => self.chunks[0].cmp(&other.chunks[0]),
                    result => result,
                },
                result => result,
            },
            result => result,
        }
    }
}
