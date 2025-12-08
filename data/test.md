# Project Aether: System Architecture & Deployment Guide

## 1. Executive Summary

**Project Aether** is a next-generation distributed storage engine designed for high-frequency trading data. Unlike traditional SQL databases, Aether utilizes a **Hyper-Graph structure** to reduce query latency by 40%. The system is currently in **Beta Phase (v0.9.4)** and is strictly for internal use only.

## 2. System Requirements

To deploy an Aether node, the host machine must meet the following strict requirements. Failure to meet the memory requirements will result in a `KernelPanic` during startup.

| Node Type       | CPU Cores | RAM (Minimum) | Storage Type | License Cost |
| :-------------- | :-------- | :------------ | :----------- | :----------- |
| **Scout**       | 4         | 16 GB         | SSD (SATA)   | Free         |
| **Vanguard**    | 16        | 64 GB         | NVMe Gen4    | $500/mo      |
| **Dreadnought** | 64        | 256 GB        | NVMe RAID-0  | $2,500/mo    |

> **Note:** Do not attempt to run the _Dreadnought_ configuration on Windows Server. It is currently optimized for Linux Kernel 5.15+ only.

## 3. Installation Protocol

Aether does not use standard package managers like `apt` or `yum`. It must be compiled from source to optimize for the specific CPU instruction set (AVX-512).

### Step 3.1: Clone and Prepare

Run the following commands in your terminal. Ensure you have `Rust` and `Cargo` installed.

```bash
# Clone the repository (Internal Network Only)
git clone ssh://git.corp.local/aether-core.git

# Enter directory
cd aether-core

# Build release binary with optimization
cargo build --release --features "avx512,async-runtime"
```

### Step 3.2: Configuration

Create a `config.toml` file in the root directory. **Warning:** The `max_connections` parameter must never exceed 10,000 on a single node without enabling "Sharding Mode".

## 4. Known Issues & Troubleshooting

### Error Code: AE-404 (Data Fragmentation)

If you see the error `AE-404: Shard detached`, it means the network latency exceeded **50ms**.

- **Cause:** Usually caused by aggressive garbage collection in the Java middleware.
- **Solution:** Increase the heartbeat interval in settings:
  `set_heartbeat(500ms)`

### The "Ghost Write" Anomaly

In version v0.9.2, there was a bug where data written during a leap second was not persisted to disk. This has been patched in v0.9.4, but legacy nodes must be manually updated.

---

## 5. FAQ

**Q: Can Aether connect to legacy SQL databases?**
A: Yes, but only via the `Bridge-SQL` adapter. Direct connections are not supported to prevent schema corruption.

**Q: What is the maximum throughput?**
A: A single _Vanguard_ node can handle approximately **1.2 million transactions per second (TPS)** under ideal laboratory conditions. Real-world performance is typically 15-20% lower.

---

### 🧪 Test Questions for your RAG:

Once you index this file, try asking these questions to see if the Markdown parsing works well:

1.  **Table Lookup:** _"How much RAM is required for a Vanguard node and what does it cost?"_
    - _Goal:_ See if it reads the table row correctly without mixing it with "Scout" or "Dreadnought".
2.  **Code/Technical:** _"What command do I use to build the project?"_
    - _Goal:_ See if it retrieves the `cargo build` command from the code block.
3.  **Specific Detail:** _"What causes the AE-404 error?"_
    - _Goal:_ See if it connects the error code to "network latency" and "garbage collection".
4.  **Reasoning:** _"Is Aether compatible with Windows Server?"_
    - _Goal:_ See if it finds the "Note" block in Section 2 which mentions the Dreadnought limitation.
