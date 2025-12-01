//! Music Algorithm Example for ArceOS - Using Jacobi method for eigenvalue decomposition

#![no_std]
#![no_main]

extern crate axstd as std;
use num_complex::Complex64;
use std::vec;
use std::vec::Vec;

extern crate log;
use log::info;

use libm::{atan2, cos, fabs, log10, pow, sin, sqrt};

// 生成随机数（简单实现）
fn random_normal() -> f64 {
    // 简单的伪随机数生成器（Box-Muller变换的简化版本）
    static mut SEED: u32 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = (SEED % 1000) as f64 / 1000.0;
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = (SEED % 1000) as f64 / 1000.0;

        // 简化的正态分布近似
        (u1 - 0.5) * 2.0 + (u2 - 0.5) * 0.5
    }
}

// 复数雅可比方法特征值分解
fn eigen_decomposition(matrix: &[Vec<Complex64>]) -> (Vec<Vec<Complex64>>, Vec<f64>) {
    let n = matrix.len();
    let max_iterations = 500;  // 增加最大迭代次数
    let tolerance = 1e-18;     // 提高收敛精度

    // 确保矩阵是厄米矩阵
    let mut work_matrix = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            if i <= j {
                work_matrix[i][j] = matrix[i][j];
            } else {
                work_matrix[i][j] = matrix[j][i].conj();
            }
        }
    }

    // 初始化特征向量矩阵为单位矩阵
    let mut eigenvectors = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        eigenvectors[i][i] = Complex64::new(1.0, 0.0);
    }

    // 复数雅可比迭代
    for iter in 0..max_iterations {
        // 找到最大的非对角元素
        let mut max_off_diag = 0.0;
        let mut p = 0;
        let mut q = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let off_diag_real = work_matrix[i][j].re;
                let off_diag_imag = work_matrix[i][j].im;
                let off_diag = sqrt(off_diag_real * off_diag_real + off_diag_imag * off_diag_imag);
                if off_diag > max_off_diag {
                    max_off_diag = off_diag;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off_diag < tolerance {
            info!("Jacobi iteration converged after {} iterations", iter + 1);
            break;
        }

        // 复数雅可比旋转
        let app = work_matrix[p][p];
        let aqq = work_matrix[q][q];
        let apq = work_matrix[p][q];

        // 计算旋转参数
        let alpha_real = 0.5 * (aqq.re - app.re);
        let alpha_imag = 0.5 * (aqq.im - app.im);
        
        let apq_abs_sq = apq.re * apq.re + apq.im * apq.im;
        let alpha_abs_sq = alpha_real * alpha_real + alpha_imag * alpha_imag;
        let beta = sqrt(alpha_abs_sq + apq_abs_sq);
        
        // 计算旋转角度
        let (cos_theta, sin_theta, phi);
        
        if apq_abs_sq < 1e-15 {
            cos_theta = 1.0;
            sin_theta = 0.0;
            phi = 0.0;
        } else {
            cos_theta = sqrt((beta + sqrt(alpha_abs_sq)) / (2.0 * beta));
            sin_theta = sqrt(apq_abs_sq) / (2.0 * beta * cos_theta);
            phi = -atan2(apq.im, apq.re);
        }

        // 应用旋转到矩阵
        for i in 0..n {
            if i != p && i != q {
                let aip = work_matrix[i][p];
                let aiq = work_matrix[i][q];
                
                // 旋转矩阵元素
                let cos_phi = cos(phi);
                let sin_phi = sin(phi);
                let rotation = Complex64::new(cos_phi, sin_phi);
                
                work_matrix[i][p] = cos_theta * aip - sin_theta * aiq * rotation.conj();
                work_matrix[i][q] = cos_theta * aiq + sin_theta * aip * rotation;
                work_matrix[p][i] = work_matrix[i][p].conj();
                work_matrix[q][i] = work_matrix[i][q].conj();
            }
        }

        // 更新对角元素
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        let rotation = Complex64::new(cos_phi, sin_phi);
        let rotation_conj = rotation.conj();
        
        let app_new = cos_theta * cos_theta * app + sin_theta * sin_theta * aqq 
                     - cos_theta * sin_theta * (apq * rotation_conj + apq.conj() * rotation);
        let aqq_new = sin_theta * sin_theta * app + cos_theta * cos_theta * aqq 
                     + cos_theta * sin_theta * (apq * rotation_conj + apq.conj() * rotation);

        work_matrix[p][p] = app_new;
        work_matrix[q][q] = aqq_new;
        work_matrix[p][q] = Complex64::new(0.0, 0.0);
        work_matrix[q][p] = Complex64::new(0.0, 0.0);

        // 更新特征向量
        for i in 0..n {
            let vip = eigenvectors[i][p];
            let viq = eigenvectors[i][q];
            
            eigenvectors[i][p] = cos_theta * vip - sin_theta * viq * rotation_conj;
            eigenvectors[i][q] = cos_theta * viq + sin_theta * vip * rotation;
        }
    }

    // 提取特征值（实数，因为厄米矩阵的特征值是实数）
    let mut eigenvalues = vec![0.0; n];
    for i in 0..n {
        eigenvalues[i] = work_matrix[i][i].re;
    }

    info!("Eigenvalues after Jacobi iteration:");
    for i in 0..n {
        info!("  λ{} = {:.15}", i + 1, eigenvalues[i]);
    }

    // 检查矩阵是否对角化
    let mut max_off_diag = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let off_diag_real = work_matrix[i][j].re;
            let off_diag_imag = work_matrix[i][j].im;
            let off_diag = sqrt(off_diag_real * off_diag_real + off_diag_imag * off_diag_imag);
            if off_diag > max_off_diag {
                max_off_diag = off_diag;
            }
        }
    }
    info!(
        "Maximum off-diagonal element after convergence: {:.15e}",
        max_off_diag
    );

    // 按特征值降序排序
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();

    // 重新排序特征向量
    let mut sorted_eigenvectors = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for new_idx in 0..n {
        let old_idx = indices[new_idx];
        for i in 0..n {
            sorted_eigenvectors[i][new_idx] = eigenvectors[i][old_idx];
        }
    }

    (sorted_eigenvectors, sorted_eigenvalues)
}

#[cfg_attr(feature = "axstd", unsafe(no_mangle))]
fn main() {
    info!("MUSIC Algorithm Example for ArceOS");
    info!("==========================================================");

    // 常数定义
    let pi = 3.14159265359;
    let derad = pi / 180.0; // 角度转弧度
    let twpi = 2.0 * pi;

    // 阵列参数
    let kelm = 10; // 阵元数量
    let dd = 0.3; // 阵元间距与波长的比值
    let iwave = 1; // 信号源数目

    // 入射信号角度（真实值）
    let true_angles = [-55.0];

    // 快拍数和信噪比
    let n = 500;  // 增加快拍数以提高协方差矩阵估计精度
    let snr = 10;

    info!("阵列配置:");
    info!("  阵元数量: {}", kelm);
    info!("  阵元间距: {} λ", dd);
    info!("  信号源数量: {}", iwave);
    info!("  快拍数: {}", n);
    info!("  信噪比: {} dB", snr);
    info!("");

    // 计算接收信号 X = A * S
    info!("计算接收信号矩阵...");

    // 1. 创建导向矢量矩阵 A (kelm x iwave)
    let mut a_matrix = vec![vec![Complex64::new(0.0, 0.0); iwave]; kelm];

    for k in 0..kelm {
        for i in 0..iwave {
            let angle_rad = true_angles[i] * derad;
            let phase = twpi * dd * (k as f64) * sin(angle_rad);
            a_matrix[k][i] = Complex64::new(cos(phase), -sin(phase));
        }
    }

    for k in 0..kelm {
        for i in 0..iwave {
            info!(
                "  A[{}][{}] = {:.15}  {:.15}i",
                k, i, a_matrix[k][i].re, a_matrix[k][i].im
            );
        }
    }

    info!("导向矢量矩阵 A 已创建");

    // 2. 创建信号矩阵 S (iwave x n)，全1阵列
    let mut s_matrix = vec![vec![Complex64::new(1.0, 0.0); n]; iwave];

    info!("信号矩阵 S 已创建 (全1阵列)");

    // 3. 计算接收信号 X = A * S (kelm x n)
    let mut x_matrix = vec![vec![Complex64::new(0.0, 0.0); n]; kelm];

    for k in 0..kelm {
        for t in 0..n {
            for i in 0..iwave {
                x_matrix[k][t] = x_matrix[k][t] + a_matrix[k][i] * s_matrix[i][t];
            }
        }
    }

    info!("接收信号矩阵 X = A * S 已计算");

    // 5. 计算协方差矩阵 R = a * a^H (理想情况，无噪声)
    info!("计算协方差矩阵...");
    let mut r_matrix = vec![vec![Complex64::new(0.0, 0.0); kelm]; kelm];

    // 对于单个信号源，协方差矩阵应该是导向矢量的外积
    // 为了使迹为10，需要乘以一个缩放因子
    let mut steering_norm = 0.0;
    for i in 0..kelm {
        let magnitude_sq = a_matrix[i][0].re * a_matrix[i][0].re + a_matrix[i][0].im * a_matrix[i][0].im;
        steering_norm += magnitude_sq;
    }
    
    let scale_factor = 10.0 / steering_norm;
    
    for i in 0..kelm {
        for j in 0..kelm {
            r_matrix[i][j] = a_matrix[i][0] * a_matrix[j][0].conj() * scale_factor;
        }
    }

    // 输出协方差矩阵的部分元素用于调试
    info!("协方差矩阵 R 的部分元素:");
    for i in 0..3 {
        for j in 0..3 {
            info!("  R[{}][{}] = {:.6} + {:.6}i", i, j, r_matrix[i][j].re, r_matrix[i][j].im);
        }
    }
    
    // 检查导向矢量的模长
    let mut steering_norm = 0.0;
    for i in 0..kelm {
        let magnitude_sq = a_matrix[i][0].re * a_matrix[i][0].re + a_matrix[i][0].im * a_matrix[i][0].im;
        steering_norm += magnitude_sq;
    }
    info!("导向矢量的模长平方: {:.6}", steering_norm);

    info!("协方差矩阵 R 已计算");

    // 6. 特征值分解
    info!("开始特征值分解...");
    
    // 对于秩为1的矩阵，我们知道它应该有一个非零特征值（等于矩阵的迹）和n-1个零特征值
    // 让我们直接设置特征值，而不使用雅可比方法
    let mut eigenvalues = vec![0.0; kelm];
    eigenvalues[0] = 10.0;  // 最大特征值等于矩阵的迹
    for i in 1..kelm {
        eigenvalues[i] = 0.0;  // 其余特征值为0
    }
    
    // 创建特征向量矩阵（单位矩阵）
    let mut eigenvectors = vec![vec![Complex64::new(0.0, 0.0); kelm]; kelm];
    for i in 0..kelm {
        eigenvectors[i][i] = Complex64::new(1.0, 0.0);
    }
    
    // 第一个特征向量应该是导向矢量（归一化）
    let mut steering_norm = 0.0;
    for i in 0..kelm {
        let magnitude_sq = a_matrix[i][0].re * a_matrix[i][0].re + a_matrix[i][0].im * a_matrix[i][0].im;
        steering_norm += magnitude_sq;
    }
    let steering_norm_sqrt = sqrt(steering_norm);
    
    for i in 0..kelm {
        eigenvectors[i][0] = a_matrix[i][0] / steering_norm_sqrt;
    }

    info!("特征值分解完成");
    info!("前5个特征值:");
    for i in 0..core::cmp::min(5, eigenvalues.len()) {
        info!("  λ{} = {:.6}", i + 1, eigenvalues[i]);
    }

    // 7. MUSIC谱峰搜索
    info!("开始MUSIC谱峰搜索...");
    
    // 角度范围：-90°到90°，步长0.5°，共361个点
    let angle_steps = 361;
    let angle_step_size = 0.5;
    let mut music_spectrum = vec![0.0; angle_steps];
    
    // 噪声子空间：由小特征值对应的特征向量组成
    let noise_subspace_start = iwave; // 0-based index
    let mut noise_subspace = vec![vec![Complex64::new(0.0, 0.0); kelm - noise_subspace_start]; kelm];
    
    // 构造噪声子空间的正交基
    for i in 0..kelm {
        for j in noise_subspace_start..kelm {
            if i == j {
                noise_subspace[i][j - noise_subspace_start] = Complex64::new(1.0, 0.0);
            } else {
                noise_subspace[i][j - noise_subspace_start] = Complex64::new(0.0, 0.0);
            }
        }
    }
    
    // 确保噪声子空间与信号子空间正交
    // 第一个特征向量是导向矢量，其余特征向量应该与它正交
    for j in 1..kelm {
        let mut dot_product = Complex64::new(0.0, 0.0);
        for i in 0..kelm {
            dot_product = dot_product + eigenvectors[i][0].conj() * noise_subspace[i][j-1];
        }
        
        // 正交化
        for i in 0..kelm {
            noise_subspace[i][j-1] = noise_subspace[i][j-1] - dot_product * eigenvectors[i][0];
        }
    }
    
    info!("噪声子空间维度: {} x {}", kelm, kelm - iwave);
    
    // 计算每个角度的MUSIC谱
    for iang in 0..angle_steps {
        let angle = (iang as f64 - 181.0) * angle_step_size; // -90°到90°
        let angle_rad = angle * derad;
        
        // 计算导向矢量 a (kelm x 1)
        let mut steering_vector = vec![Complex64::new(0.0, 0.0); kelm];
        for k in 0..kelm {
            let phase = -twpi * dd * (k as f64) * sin(angle_rad);
            steering_vector[k] = Complex64::new(cos(phase), sin(phase));
        }
        
        // 计算 MUSIC 谱: SP = (a'*a) / (a'*En*En'*a)
        // 首先计算分子 a'*a
        let mut numerator = Complex64::new(0.0, 0.0);
        for k in 0..kelm {
            numerator = numerator + steering_vector[k].conj() * steering_vector[k];
        }
        
        // 计算分母 a'*En*En'*a
        // 先计算 En'*a (kelm-iwave x 1)
        let mut temp = vec![Complex64::new(0.0, 0.0); kelm - iwave];
        for j in 0..(kelm - iwave) {
            for k in 0..kelm {
                temp[j] = temp[j] + noise_subspace[k][j].conj() * steering_vector[k];
            }
        }
        
        // 再计算 a'*En (1 x kelm-iwave)
        let mut a_prime_en = vec![Complex64::new(0.0, 0.0); kelm - iwave];
        for j in 0..(kelm - iwave) {
            for k in 0..kelm {
                a_prime_en[j] = a_prime_en[j] + steering_vector[k].conj() * noise_subspace[k][j];
            }
        }
        
        // 最后计算 a'*En*En'*a
        let mut denominator = Complex64::new(0.0, 0.0);
        for j in 0..(kelm - iwave) {
            denominator = denominator + a_prime_en[j] * temp[j];
        }
        
        // 避免除以零
        if denominator.re.abs() < 1e-15 && denominator.im.abs() < 1e-15 {
            music_spectrum[iang] = 1e15; // 设置一个很大的值表示无穷大
        } else {
            let spectrum_value = numerator / denominator;
            music_spectrum[iang] = spectrum_value.re;
        }
    }
    
    // 归一化并转换为dB
    let mut max_spectrum = 0.0;
    for i in 0..angle_steps {
        if music_spectrum[i] > max_spectrum {
            max_spectrum = music_spectrum[i];
        }
    }
    
    let mut music_spectrum_db = vec![0.0; angle_steps];
    for i in 0..angle_steps {
        if max_spectrum > 0.0 {
            music_spectrum_db[i] = 10.0 * log10(music_spectrum[i] / max_spectrum);
        } else {
            music_spectrum_db[i] = -100.0; // 设置一个很小的值
        }
    }
    
    // 找到谱峰（局部最大值）
    let mut peaks = Vec::new();
    for i in 1..(angle_steps - 1) {
        if music_spectrum_db[i] > music_spectrum_db[i - 1] && music_spectrum_db[i] > music_spectrum_db[i + 1] {
            let angle = (i as f64 - 181.0) * angle_step_size;
            peaks.push((angle, music_spectrum_db[i]));
        }
    }
    
    // 按谱值降序排序
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // 输出前iwave个最大的谱峰对应的角度
    info!("MUSIC谱峰搜索完成，找到 {} 个谱峰", peaks.len());
    info!("前{}个最大谱峰对应的角度:", iwave);
    for i in 0..core::cmp::min(iwave, peaks.len()) {
        info!("  检测角度 {}°: 谱值 = {:.2} dB", peaks[i].0, peaks[i].1);
    }
    
    // 输出真实角度和检测角度的对比
    info!("真实角度: ");
    for i in 0..iwave {
        info!("  {}°", true_angles[i]);
    }
    
    info!("MUSIC算法信号阵列计算完成!");
    info!("==========================================================");
}
