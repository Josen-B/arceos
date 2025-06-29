use aarch64_cpu::registers::{Readable, Writeable};
use core::ptr::NonNull;
use tock_registers::{register_structs, registers::ReadWrite};

register_structs! {
    pub WatchDogRegs {
        (0x0000 => pub wdt_wrr: ReadWrite<u32>),      // 更新寄存器
        (0x0004 => __reserved0),
        (0x0fcc => pub wdt_w_iidr: ReadWrite<u32>),   // 接口身份试别寄存器
        (0x0fd0 => __reserved1),
        (0x1000 => pub wdt_wcs: ReadWrite<u32>),      // 控制和状态寄存器
        (0x1004 => __reserved2),
        (0x1008 => pub wdt_wor: ReadWrite<u32>),      // 清除寄存器
        (0x100c => __reserved3),
        (0x1010 => pub wdt_wcvl: ReadWrite<u32>),     // 比较值低32位寄存器
        (0x1014 => pub wdt_wcvh: ReadWrite<u32>),     // 比较值高32位寄存器
        (0x1018 => @END),                             // 结束寄存器
    }
}

pub struct WatchDog {
    base: NonNull<WatchDogRegs>,
}

unsafe impl Send for WatchDog {}
unsafe impl Sync for WatchDog {}

impl WatchDog {
    pub const fn new(base: *mut u8) -> Self {
        Self {
            base: NonNull::new(base).unwrap().cast(),
        }
    }

    const fn regs(&self) -> &WatchDogRegs {
        unsafe { self.base.as_ref() }
    }

    // 飞腾派硬件板子上的Watchdog看门狗设备
    /// 初始化:
    //  1. 可以设置看门狗计数超时值WOR，未设置则默认为1s超时;若设置超时时间,
    //     通过写WOR寄存器会直接将当前syscnt+WOR寄存器储存的值更新到WCV寄存器中;
    //  2. 通过WCS寄存器Enable Watchdog，同时写此WCS寄存器也会有喂狗的效果
    pub fn init(&mut self) {
        self.regs().wdt_wcs.set(0x1);
    }

    // 检查是否初始化
    pub fn is_init(&self) -> bool {
        if self.regs().wdt_wcs.get() & 0x1 != 0 {
            debug!("WatchDog is initialized");
            let info = self.regs().wdt_w_iidr.get();
            debug!(
                "WatchDog ID: version={}, continuation_code={}, identity_code={}",
                (info >> 16) as u16,      // 版本号
                (info >> 8 & 0xFF) as u8, // 续码
                (info & 0xFF) as u8       // 身份码
            );
            return true;
        } else {
            warn!("WatchDog is not initialized");
            return false;
        }
    }

    // 设置看门狗计数超时值WOR
    pub fn set_timeout(&mut self, timeout: u32) {
        self.regs().wdt_wor.set(timeout);
    }

    // 启动看门狗
    pub fn start(&mut self) {
        self.regs().wdt_wcs.set(0x1);
    }

    // 停止看门狗
    pub fn stop(&mut self) {
        self.regs().wdt_wcs.set(0x0);
    }

    // 喂狗操作--写WRR寄存器
    // 何时超时: sys_cnt的计数值大于当前WCV寄存器存储的比较值。
    // 比较值wcv=当前sys_cnt+计数超时值wor
    // 1. ws0一次超时,控制器报中断,后需要进行喂狗，若在计数时间内(WOR)无喂狗操作则会二次超时;
    // 2. ws1二次超时,控制器发起复位。
    pub fn feed(&mut self) {
        // 喂狗操作--写WRR寄存器
        debug!("Feeding WatchDog");
        self.regs().wdt_wrr.set(0x1);
    }
}
