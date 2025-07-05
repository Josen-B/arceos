use crate::mem::{PhysAddr, phys_to_virt};
use aarch64_cpu::registers::{Readable, Writeable};
use core::ptr::NonNull;
use kspin::SpinNoIrq;
use tock_registers::{
    register_bitfields, register_structs,
    registers::{ReadOnly, ReadWrite, WriteOnly},
};

pub const BAUD_RATE : u32 = 115200; // 波特率
pub const CLK_RATE : u32 = 100_000_000; // 时钟频率

const UART_BASE: PhysAddr = pa!(0x2800_E000);

pub static UART: SpinNoIrq<Uart> =
    SpinNoIrq::new(Uart::new(phys_to_virt(UART_BASE).as_mut_ptr()));

register_structs! {
    UartRegs {
        (0x000 => uartdr: ReadWrite<u32, UARTDR::Register>),
        (0x004 => pub uartecr: ReadOnly<u32>),
        (0x008 => _reserved0),
        (0x018 => uartfr: ReadOnly<u32, UARTFR::Register>),
        (0x01c => _reserved1),
        (0x024 => uartibrd: ReadWrite<u32, UARTIBRD::Register>),
        (0x028 => uartfbrd: ReadWrite<u32, UARTFBRD::Register>),
        (0x02c => uartlcrh: ReadWrite<u32, UARTLCRH::Register>),
        (0x030 => uartcr: ReadWrite<u32,UARTCR::Register>),
        (0x034 => uartifls: ReadWrite<u32, UARTIFLS::Register>),
        (0x038 => uartimsc: ReadWrite<u32, UARTIMSC::Register>),
        (0x03c => uartris: ReadOnly<u32, UARTRIS::Register>),
        (0x040 => uartmis: ReadOnly<u32>),
        (0x044 => uarticr: WriteOnly<u32, UARTICR::Register>),
        (0x048 => pub uartdmacr: ReadWrite<u32>),
        (0x04c => @END),
    }
}

register_bitfields![u32,
    pub UARTDR [
        data OFFSET(0) NUMBITS(8) [], // 数据
        fe OFFSET(8) NUMBITS(1) [],   // 奇偶校验错误
        pe OFFSET(9) NUMBITS(1) [],  // 奇偶校验错误
        be OFFSET(10) NUMBITS(1) [],  // 帧错误
        oe OFFSET(11) NUMBITS(1) []    // 溢出错误
    ],
    pub UARTFR [
        cts OFFSET(0) NUMBITS(1) [],   // CTS信号
        dsr OFFSET(1) NUMBITS(1) [], // DSR信号
        dcd OFFSET(2) NUMBITS(1) [], // DCD信号
        busy OFFSET(3) NUMBITS(1) [], // 发送忙
        rxfe OFFSET(4) NUMBITS(1) [],  // 接收FIFO为空
        txff OFFSET(5) NUMBITS(1) [],  // 发送FIFO满
        rxff OFFSET(6) NUMBITS(1) [],  // 接收FIFO满
        txfe OFFSET(7) NUMBITS(1) [],  // 发送FIFO为空
        ri OFFSET(8) NUMBITS(1) []     // RI信号
    ],
    pub UARTCR [
        uarten OFFSET(0) NUMBITS(1) [], // UART使能
        txe OFFSET(8) NUMBITS(1) [],    // 发送使能
        rxe OFFSET(9) NUMBITS(1) [],    // 接收使能
        dtr OFFSET(10) NUMBITS(1) [],   // DTR信号
        rts OFFSET(11) NUMBITS(1) []     // RTS信号
    ],
    pub UARTIBRD [
        integer OFFSET(0) NUMBITS(16) [] // 整数部分
    ],
    pub UARTFBRD [
        integer OFFSET(0) NUMBITS(8) [], // 整数部分
        fraction OFFSET(8) NUMBITS(6) [] // 小数部分
    ],
    pub UARTLCRH [
        wlen OFFSET(0) NUMBITS(2) [], // 字长
        stp2 OFFSET(2) NUMBITS(1) [], // 双停止位
        pen OFFSET(3) NUMBITS(1) [],  // 奇偶校验使能
        eps OFFSET(4) NUMBITS(1) [],  // 奇偶校验选择
        brk OFFSET(5) NUMBITS(1) [],  // 发送中断
        fpen OFFSET(6) NUMBITS(1) []   // 硬件流控制使能
    ],
    pub UARTIMSC [
        rxim OFFSET(4) NUMBITS(1) [], // 接收中断使能
        txim OFFSET(5) NUMBITS(1) [], // 发送中断使能
        rtim OFFSET(6) NUMBITS(1) [], // 接收超时中断使能
        feim OFFSET(7) NUMBITS(1) [], // 奇偶校验错误中断使能
        peim OFFSET(8) NUMBITS(1) [], // 奇偶校验错误中断使能
        beim OFFSET(9) NUMBITS(1) [], // 帧错误中断使能
        oem OFFSET(10) NUMBITS(1) []  // 溢出错误中断使能
    ],
    pub UARTIFLS [
        txif OFFSET(0) NUMBITS(3) [], // 发送FIFO阈值
        rxif OFFSET(3) NUMBITS(3) []  // 接收FIFO阈值
    ],
    pub UARTRIS [
        rxis OFFSET(4) NUMBITS(1) [], // 接收中断状态
        txis OFFSET(5) NUMBITS(1) [], // 发送中断状态
        rtis OFFSET(6) NUMBITS(1) [], // 接收超时中断状态
        feis OFFSET(7) NUMBITS(1) [], // 奇偶校验错误中断状态
        peis OFFSET(8) NUMBITS(1) [], // 奇偶校验错误中断状态
        beis OFFSET(9) NUMBITS(1) [], // 帧错误中断状态
        oem OFFSET(10) NUMBITS(1) []  // 溢出错误中断状态
    ],
    pub UARTICR [
        rxim OFFSET(4) NUMBITS(1) [], // 清除接收中断
        txim OFFSET(5) NUMBITS(1) [], // 清除发送中断
        rtim OFFSET(6) NUMBITS(1) [], // 清除接收超时中断
        feim OFFSET(7) NUMBITS(1) [], // 清除奇偶校验错误中断
        peim OFFSET(8) NUMBITS(1) [], // 清除奇偶校验错误中断
        beim OFFSET(9) NUMBITS(1) [], // 清除帧错误中断
        oem OFFSET(10) NUMBITS(1) []  // 清除溢出错误中断
    ]
];

pub struct Uart {
    base: NonNull<UartRegs>,
}

unsafe impl Send for Uart {}
unsafe impl Sync for Uart {}

impl Uart {
    pub const fn new(base: *mut u8) -> Self {
        Self {
            base: NonNull::new(base).unwrap().cast(),
        }
    }

    pub fn init(&self) {
        let uart = unsafe { self.base.as_ref() };
        // 关闭 UART
        uart.uartcr.write(UARTCR::uarten::CLEAR);
        // 设置波特率
        let integer_part = CLK_RATE / (16 * BAUD_RATE);
        let fraction_part = ((CLK_RATE % (16 * BAUD_RATE)) * 64 / (16 * BAUD_RATE)) as u8;
        info!("integer_part is {}, fraction_part is {}", integer_part, fraction_part);
        uart.uartibrd.set(integer_part);
        uart.uartfbrd.set(fraction_part as u32);
        // 使能fifo
        uart.uartifls.set(0x20);
        // 配置 UART
        info!("configuring UART");
        uart.uartlcrh.set(0x70); // 8位数据, 无奇偶校验, 1位停止位, FIFOs使能
        uart.uartcr.write(
            UARTCR::uarten::SET
                + UARTCR::txe::SET
                + UARTCR::rxe::SET
                + UARTCR::dtr::SET
                + UARTCR::rts::SET,
        );
    }

    // 发送数据
    pub fn send(&self, data: u8) {
        let uart = unsafe { self.base.as_ref() };
        if uart.uartfr.read(UARTFR::txfe) == 0 {
            info!("FIFO not empty, waiting to send data");
        }
        uart.uartdr.set(data as u32);
    }

    // 接收数据
    pub fn receive(&self) -> u8 {
        let uart = unsafe { self.base.as_ref() };
        if uart.uartfr.read(UARTFR::rxfe) != 0 {
            warn!("FIFO is empty, no data to receive");
            return 0; // 或者返回一个错误值
        }
        uart.uartdr.get() as u8
    }
}
