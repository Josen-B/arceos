use super::watchdog::*;
use crate::mem::phys_to_virt;
use crate::platform::aarch64_common::gic::{WATCHDOG_IRQ_NUM, register_handler};
use aarch64_cpu::asm::nop;
use kspin::SpinNoIrq;
use memory_addr::PhysAddr;

const WATCHDOG_BASE: PhysAddr = pa!(0x2804_0000); // WatchDog base address

static WATCHDOG: SpinNoIrq<WatchDog> =
    SpinNoIrq::new(WatchDog::new(phys_to_virt(WATCHDOG_BASE).as_mut_ptr()));

pub fn watchdog_example() {
    // set interrupt enable
    crate::irq::set_enable(WATCHDOG_IRQ_NUM, true);
    // register handler
    register_handler(WATCHDOG_IRQ_NUM, handle_wdt_irq);
    // 初始化看门狗
    info!("Initializing WatchDog");
    WATCHDOG.lock().init();
    // 检查是否初始化
    info!("Checking if WatchDog is initialized");
    if !WATCHDOG.lock().is_init() {
        return;
    }
    // 设置看门狗计数超时值WOR
    // info!("Setting WatchDog timeout");
    // WATCHDOG.lock().set_timeout(0x10000);
    // 启动看门狗
    info!("Starting WatchDog");
    WATCHDOG.lock().start();
    // 喂狗操作
    for _ in 0..5 {
        // 喂狗操作
        WATCHDOG.lock().feed();
        // 模拟一些工作
        nop();
    }
    // 取消看门狗
    // info!("Disabling WatchDog");
    // WATCHDOG.lock().stop();
}

pub fn handle_wdt_irq() {
    debug!("WatchDog IRQ triggered");
}
