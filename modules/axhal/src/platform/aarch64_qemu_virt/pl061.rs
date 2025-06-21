use crate::misc::terminate;
use crate::platform::aarch64_common::gic::{GPIO_IRQ_NUM, register_handler};
use aarch64_cpu::registers::{Readable, Writeable};
use tock_registers::{
    register_bitfields, register_structs,
    registers::{ReadOnly, ReadWrite, WriteOnly},
};

pub const PLO61REGS: *mut PL061Regs =
    (axconfig::plat::PHYS_VIRT_OFFSET + 0x0903_0000) as *mut PL061Regs;

register_bitfields![u32,
// GPIO_DATA
    pub GPIO_DATA [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
// GPIO_DIR
    pub GPIO_DIR [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
// GPIO_IS
    pub GPIO_IS [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
    // GPIO_IBE
    pub GPIO_IBE [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
    // GPIO_IEV
    pub GPIO_IEV [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
    // GPIO_IE
    pub GPIO_IE [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
    // GPIO_RIS
    pub GPIO_RIS [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
    // GPIO_MIS
    pub GPIO_MIS [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
    // GPIO_IC
    pub GPIO_IC [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];
register_bitfields![u32,
    // GPIO_AFSEL
    pub GPIO_AFSEL [
        pin0 OFFSET(0) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin1 OFFSET(1) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin2 OFFSET(2) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin3 OFFSET(3) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin4 OFFSET(4) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin5 OFFSET(5) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin6 OFFSET(6) NUMBITS(1) [
            clear = 0,
            set = 1
        ],
        pin7 OFFSET(7) NUMBITS(1) [
            clear = 0,
            set = 1
        ]
    ]
];

register_structs! {
    pub PL061Regs {
        (0x000 => pub gpio_data: ReadWrite<u32>),
        (0x004 => __reserved0),
        (0x400 => pub gpio_dir: ReadWrite<u32, GPIO_DIR::Register>),
        (0x404 => pub gpio_is: ReadWrite<u32, GPIO_IS::Register>),
        (0x408 => pub gpio_ibe: ReadWrite<u32, GPIO_IBE::Register>),
        (0x40c => pub gpio_iev: ReadWrite<u32, GPIO_IEV::Register>),
        (0x410 => pub gpio_ie: ReadWrite<u32, GPIO_IE::Register>),
        (0x414 => pub gpio_ris: ReadOnly<u32, GPIO_RIS::Register>),
        (0x418 => pub gpio_mis: ReadOnly<u32, GPIO_MIS::Register>),
        (0x41c => pub gpio_ic: WriteOnly<u32, GPIO_IC::Register>),
        (0x420 => pub gpio_afsel: ReadWrite<u32, GPIO_AFSEL::Register>),
        (0x424 => @END),
    }
}

/// init gpio interrupt
pub fn gpio_init() {
    info!("Initializing GPIO interrupt");
    // set interrupt enable
    crate::irq::set_enable(GPIO_IRQ_NUM, true);
    // register handler
    register_handler(GPIO_IRQ_NUM, handle_gpio_irq);
    info!("GPIO IRQ handler registered");
    unsafe {
        let pl061: &PL061Regs = &mut *PLO61REGS;
        pl061.gpio_ie.write(GPIO_IE::pin3::set);
    }
}

/// handle gpio interrupt and power down
pub fn handle_gpio_irq() {
    unsafe {
        let pl061: &PL061Regs = &mut *PLO61REGS;
        pl061.gpio_ic.set(pl061.gpio_ie.get());
    }
    info!("Powerdown now");
    terminate();
}
