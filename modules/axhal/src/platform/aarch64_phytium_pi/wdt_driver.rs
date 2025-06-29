use super::watchdog::*;
use aarch64_cpu::asm::nop;
use kspin::SpinNoIrq;
use crate::mem::phys_to_virt;
use memory_addr::PhysAddr;
use crate::platform::aarch64_common::gic::{WATCHDOG_IRQ_NUM, register_handler};

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
    if !WATCHDOG.lock().is_init() {
        return;
    }
    // 喂狗操作
    for _ in 0..5 {
        // 喂狗操作
        WATCHDOG.lock().feed();
        // 模拟一些工作
        nop();
    }
}

pub fn handle_wdt_irq() {
    info!("WatchDog IRQ triggered");
}