#![cfg_attr(feature = "axstd", no_std)]
#![cfg_attr(feature = "axstd", no_main)]

#[cfg(feature = "axstd")]
use axstd::println;

use axstd::console::UART;

#[cfg_attr(feature = "axstd", unsafe(no_mangle))]
fn main() {
    println!("UART example started");
    let mut uart = UART.lock();
    uart.init();
    println!("UART initialized, ready to send data.\n");
    for i in 1..10 {
        uart.send(i as u8);
        println!("Sent byte: {}\n", i);
        let received = uart.receive();
        if received != 0 {
            println!("Received byte: {}\n", received);
        }
    }
    println!("UART example finished");
}
