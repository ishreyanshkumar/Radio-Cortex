use std::process::Command;

fn main() {
    println!("--- Radio-Cortex System Monitor (Rust) ---");
    
    let output = Command::new("free")
        .arg("-h")
        .output()
        .expect("failed to execute process");

    println!("Memory Usage:\n{}", String::from_utf8_lossy(&output.stdout));
    
    let load = Command::new("uptime")
        .output()
        .expect("failed to execute process");
        
    println!("System Load:\n{}", String::from_utf8_lossy(&load.stdout));
}
