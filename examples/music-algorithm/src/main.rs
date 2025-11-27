//! Music Algorithm Example for ArceOS - Using Jacobi method for eigenvalue decomposition

#![no_std]
#![no_main]

extern crate axstd as std;
use std::vec::Vec;
use std::vec;
use core::prelude::rust_2024::derive;

extern crate log;
use log::info;

use libm::{sin, cos, sqrt, exp, log as ln, atan2};

// 复数结构体
#[derive(Debug, Clone, Copy)]
struct MyComplex {
    real: f64,
    imag: f64,
}

impl MyComplex {
    fn new(real: f64, imag: f64) -> MyComplex {
        MyComplex { real, imag }
    }
    
    fn conj(&self) -> MyComplex {
        MyComplex {
            real: self.real,
            imag: -self.imag,
        }
    }
    
    fn add(&self, other: &MyComplex) -> MyComplex {
        MyComplex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
    
    fn mul(&self, other: &MyComplex) -> MyComplex {
        MyComplex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    fn scale(&self, scalar: f64) -> MyComplex {
        MyComplex {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }
    
    fn magnitude(&self) -> f64 {
        sqrt(self.real * self.real + self.imag * self.imag)
    }
    
    fn phase(&self) -> f64 {
        libm::atan2(self.imag, self.real)
    }
    
    fn from_polar(r: f64, theta: f64) -> MyComplex {
        MyComplex {
            real: r * cos(theta),
            imag: r * sin(theta),
        }
    }
}

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

// 添加高斯白噪声
fn add_awgn(signal: &[Vec<MyComplex>], snr_db: f64) -> Vec<Vec<MyComplex>> {
    let rows = signal.len();
    let cols = signal[0].len();
    let mut noisy_signal = vec![vec![MyComplex::new(0.0, 0.0); cols]; rows];

    // 计算信号功率
    let mut signal_power = 0.0f64;
    for i in 0..rows {
        for j in 0..cols {
            signal_power += signal[i][j].magnitude() * signal[i][j].magnitude();
        }
    }
    signal_power /= (rows * cols) as f64;

    // 计算噪声功率
    let snr_linear = exp(10.0 * ln(10.0) * snr_db / 10.0);
    let noise_power = signal_power / snr_linear;
    let noise_std = sqrt(noise_power / 2.0);

    for i in 0..rows {
        for j in 0..cols {
            let noise_real = random_normal() * noise_std;
            let noise_imag = random_normal() * noise_std;
            noisy_signal[i][j] = MyComplex::new(
                signal[i][j].real + noise_real,
                signal[i][j].imag + noise_imag
            );
        }
    }

    noisy_signal
}

// 矩阵乘法
fn matrix_multiply(a: &[Vec<MyComplex>], b: &[Vec<MyComplex>]) -> Vec<Vec<MyComplex>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let cols_b = b[0].len();
    let mut result = vec![vec![MyComplex::new(0.0, 0.0); cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = MyComplex::new(0.0, 0.0);
            for k in 0..cols_a {
                sum = sum.add(&a[i][k].mul(&b[k][j]));
            }
            result[i][j] = sum;
        }
    }
    result
}

// 矩阵转置共轭
fn matrix_transpose_conjugate(a: &[Vec<MyComplex>]) -> Vec<Vec<MyComplex>> {
    let rows = a.len();
    let cols = a[0].len();
    let mut result = vec![vec![MyComplex::new(0.0, 0.0); rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = a[i][j].conj();
        }
    }
    result
}

// 矩阵缩放
fn matrix_scale(a: &[Vec<MyComplex>], scalar: f64) -> Vec<Vec<MyComplex>> {
    let rows = a.len();
    let cols = a[0].len();
    let mut result = vec![vec![MyComplex::new(0.0, 0.0); cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i][j] = a[i][j].scale(scalar);
        }
    }
    result
}

// 简化的特征值分解 - 使用幂迭代法近似计算
fn eigen_decomposition(matrix: &[Vec<MyComplex>]) -> (Vec<Vec<MyComplex>>, Vec<f64>) {
    let n = matrix.len();
    let max_iterations = 100;
    let tolerance = 1e-8;
    
    // 首先计算实数协方差矩阵（厄米矩阵的实部）
    let mut real_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            real_matrix[i][j] = matrix[i][j].real;
        }
    }
    
    // 使用简化的QR算法计算特征值和特征向量
    let mut eigenvalues = vec![0.0; n];
    let mut eigenvectors = vec![vec![0.0; n]; n];
    
    // 初始化特征向量矩阵为单位矩阵
    for i in 0..n {
        eigenvectors[i][i] = 1.0;
    }
    
    // 简化的幂迭代法计算主特征值
    let mut work_matrix = real_matrix.clone();
    
    for _iter in 0..max_iterations {
        // 找到最大的非对角元素
        let mut max_off_diag = 0.0;
        let mut p = 0;
        let mut q = 0;
        
        for i in 0..n {
            for j in (i+1)..n {
                if work_matrix[i][j].abs() > max_off_diag {
                    max_off_diag = work_matrix[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        
        if max_off_diag < tolerance {
            break;
        }
        
        // Jacobi旋转（实数版本）
        let app = work_matrix[p][p];
        let aqq = work_matrix[q][q];
        let apq = work_matrix[p][q];
        
        let theta = 0.5 * atan2(2.0 * apq, aqq - app);
        let cos_theta = cos(theta);
        let sin_theta = sin(theta);
        
        // 更新矩阵
        for i in 0..n {
            if i != p && i != q {
                let aip = work_matrix[i][p];
                let aiq = work_matrix[i][q];
                work_matrix[i][p] = cos_theta * aip - sin_theta * aiq;
                work_matrix[i][q] = sin_theta * aip + cos_theta * aiq;
                work_matrix[p][i] = work_matrix[i][p];
                work_matrix[q][i] = work_matrix[i][q];
            }
        }
        
        let app_new = cos_theta * cos_theta * app + sin_theta * sin_theta * aqq - 2.0 * cos_theta * sin_theta * apq;
        let aqq_new = sin_theta * sin_theta * app + cos_theta * cos_theta * aqq + 2.0 * cos_theta * sin_theta * apq;
        
        work_matrix[p][p] = app_new;
        work_matrix[q][q] = aqq_new;
        work_matrix[p][q] = 0.0;
        work_matrix[q][p] = 0.0;
        
        // 更新特征向量
        for i in 0..n {
            let vip = eigenvectors[i][p];
            let viq = eigenvectors[i][q];
            eigenvectors[i][p] = cos_theta * vip - sin_theta * viq;
            eigenvectors[i][q] = sin_theta * vip + cos_theta * viq;
        }
    }
    
    // 提取特征值
    for i in 0..n {
        eigenvalues[i] = work_matrix[i][i];
    }
    
    info!("Eigenvalues after iteration:");
    for i in 0..n {
        info!("  λ{} = {:.6}", i+1, eigenvalues[i]);
    }
    
    // 检查矩阵是否对角化
    let mut max_off_diag = 0.0;
    for i in 0..n {
        for j in (i+1)..n {
            let off_diag = work_matrix[i][j].abs();
            if off_diag > max_off_diag {
                max_off_diag = off_diag;
            }
        }
    }
    info!("Maximum off-diagonal element after convergence: {:.6e}", max_off_diag);
    
    // 检查协方差矩阵的性质
    info!("Covariance matrix diagonal elements:");
    for i in 0..n {
        info!("  R[{}][{}] = {:.6}", i, i, matrix[i][i].real);
    }
    
    // 按特征值降序排序
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    
    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    
    // 将实数特征向量转换为复数形式并重新排序
    let mut sorted_eigenvectors = vec![vec![MyComplex::new(0.0, 0.0); n]; n];
    for new_idx in 0..n {
        let old_idx = indices[new_idx];
        for i in 0..n {
            sorted_eigenvectors[i][new_idx] = MyComplex::new(eigenvectors[i][old_idx], 0.0);
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
    let derad = pi / 180.0;  // 角度转弧度
    let twpi = 2.0 * pi;
    
    // 阵列参数
    let kelm = 10;              // 阵元数量
    let dd = 0.5;               // 阵元间距与波长的比值
    let iwave = 3;              // 信号源数目
    
    // 入射信号角度（真实值）
    let true_angles = [0.0, 55.0, 80.0];
    
    // 快拍数和信噪比
    let n = 500;
    let snr = 10;
    
    info!("阵列配置:");
    info!("  阵元数量: {}", kelm);
    info!("  阵元间距: {} λ", dd);
    info!("  信号源数量: {}", iwave);
    info!("  快拍数: {}", n);
    info!("  信噪比: {} dB", snr);
    info!("");
    
    // 构建阵元位置向量
    let mut d = Vec::new();
    for i in 0..kelm {
        d.push((i as f64) * dd);
    }
    
    // 构建信号导向矢量矩阵 A
    let mut steering_matrix = vec![vec![MyComplex::new(0.0, 0.0); iwave]; kelm];
    for sensor in 0..kelm {
        for source in 0..iwave {
            let angle_rad = true_angles[source] * derad;
            let phase = -twpi * d[sensor] * sin(angle_rad);
            steering_matrix[sensor][source] = MyComplex::from_polar(1.0, phase);
        }
    }
    
    // 生成随机信号源 S
    let mut signal_sources = vec![vec![MyComplex::new(0.0, 0.0); n]; iwave];
    for source in 0..iwave {
        for snap in 0..n {
            signal_sources[source][snap] = MyComplex::new(random_normal(), random_normal());
        }
    }
    
    // 计算接收信号 X = A * S
    let received_signal = matrix_multiply(&steering_matrix, &signal_sources);
    
    // 添加高斯白噪声
    let noisy_signal = add_awgn(&received_signal, snr as f64);
    
    // 计算协方差矩阵 Rxx = X * X' / n
    let signal_transpose = matrix_transpose_conjugate(&noisy_signal);
    let covariance = matrix_multiply(&noisy_signal, &signal_transpose);
    let covariance = matrix_scale(&covariance, 1.0 / (n as f64));
    
    info!("协方差矩阵计算完成");
    info!("协方差矩阵维度: {}x{}", covariance.len(), covariance[0].len());
    info!("协方差矩阵[0][0] = {:.4}", covariance[0][0].magnitude());
    
    // 特征值分解
    let (eigenvectors, eigenvalues) = eigen_decomposition(&covariance);
    
    // 分离噪声子空间 En (特征值较小的特征向量)
    let mut noise_subspace = vec![vec![MyComplex::new(0.0, 0.0); kelm - iwave]; kelm];
    for i in 0..kelm {
        for j in iwave..kelm {
            noise_subspace[i][j - iwave] = eigenvectors[i][j];
        }
    }
    
    // MUSIC谱计算
    let mut spectrum = vec![0.0f64; 361]; // -180度到180度，步长1度
    
    for angle_idx in 0..361 {
        let angle = (angle_idx as f64 - 180.0) / 2.0; // 与MATLAB一致，步长0.5度
        let angle_rad = angle * derad;
        
        // 构建导向矢量 a
        let mut steering_vector = vec![MyComplex::new(0.0, 0.0); kelm];
        for sensor in 0..kelm {
            let phase = -twpi * d[sensor] * sin(angle_rad);
            steering_vector[sensor] = MyComplex::from_polar(1.0, phase);
        }
        
        // 计算 MUSIC 谱: P(θ) = 1 / (a'*En*En'*a)
        // 分子恒为1，所以只需要计算分母
        let mut denominator = MyComplex::new(0.0, 0.0);
        
        // 计算 En * En' * a 的投影能量
        // 首先计算 En' * a
        let mut en_trans_a = vec![MyComplex::new(0.0, 0.0); kelm - iwave];
        for i in 0..(kelm - iwave) {
            for j in 0..kelm {
                en_trans_a[i] = en_trans_a[i].add(&noise_subspace[j][i].conj().mul(&steering_vector[j]));
            }
        }
        
        // 然后计算 ||En' * a||^2
        for i in 0..(kelm - iwave) {
            denominator = denominator.add(&en_trans_a[i].conj().mul(&en_trans_a[i]));
        }
        
        // 计算谱值（注意分母很小的时候谱值会很大）
        if denominator.magnitude() > 1e-10 {
            spectrum[angle_idx] = 1.0 / denominator.magnitude();
        } else {
            spectrum[angle_idx] = 1e10; // 设置一个很大的值表示谱峰
        }
    }
    
    // 寻找谱峰
    let mut peaks = Vec::new();
    
    // 首先找到最大谱值，用于动态设置阈值
    let max_spectrum = spectrum.iter().fold(0.0f64, |a, &b| a.max(b));
    let min_peak_height = max_spectrum * 0.1; // 使用最大值的10%作为阈值
    
    for i in 1..360 {
        if spectrum[i] > min_peak_height && 
           spectrum[i] > spectrum[i-1] && 
           spectrum[i] > spectrum[i+1] {
            let angle = (i as f64 - 180.0) / 2.0;
            peaks.push(angle);
        }
    }
    
    // 按谱值大小排序峰值
    peaks.sort_by(|&a, &b| {
        let idx_a = ((a * 2.0) + 180.0) as usize;
        let idx_b = ((b * 2.0) + 180.0) as usize;
        spectrum[idx_b].partial_cmp(&spectrum[idx_a]).unwrap()
    });
    
    // 只取前iwave个最强的峰值
    peaks.truncate(iwave);
    
    // 输出结果
    info!("");
    info!("结果:");
    info!("----------------------------------------");
    info!("真实信号角度: ");
    for (i, &angle) in true_angles.iter().enumerate() {
        info!("  信号 {}: {:6.1}°", i+1, angle);
    }
    
    info!("");
    info!("MUSIC估计角度: ");
    if peaks.is_empty() {
        info!("  未检测到明显的信号方向");
    } else {
        for (i, &angle) in peaks.iter().enumerate() {
            info!("  检测 {}: {:6.1}°", i+1, angle);
        }
    }
    
    info!("");
    info!("特征值:");
    info!("----------------------------------------");
    for (i, &value) in eigenvalues.iter().enumerate() {
        info!("  λ{} = {:.4}", i+1, value);
    }
    
    info!("");
    info!("谱峰信息:");
    info!("----------------------------------------");
    for &angle in &peaks {
        let angle_idx = ((angle * 2.0) + 180.0) as usize;
        if angle_idx < 361 {
            info!("  角度 {:6.1}°: 谱值 = {:.2e}", angle, spectrum[angle_idx]);
        }
    }
    
    info!("");
    info!("算法执行完成！");
}