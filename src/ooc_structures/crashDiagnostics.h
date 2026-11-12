#pragma once

// Crash diagnostics: on SIGSEGV / SIGBUS, prints what is needed to tell a memory error from a
// corrupted thread control block, plus the memory state of the machine, then re-raises the signal
// (default action: core dump). Call `installCrashHandler()' once, at the start of `main'.
//
// Interpretation of the printed line:
//  - code 1 (SEGV_MAPERR) / 2 (SEGV_ACCERR): a real invalid access at `addr', by the instruction at `rip'
//  - code 128 (SI_KERNEL) with addr 0: raised by the kernel. If `rseq_cs' is not 0, or a cpu value is
//    larger than the number of cores, the thread control block of this thread was overwritten
//  - MemAvailable / SwapFree close to 0: the machine was running out of memory

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/rseq.h>
#include <ucontext.h>
#include <unistd.h>

namespace crash_diagnostics {

inline void writeMatchingLines(const char* path, const char* const* prefixes, int nb_prefixes){
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if(fd < 0){ return; }
    char data[8192];
    ssize_t n = read(fd, data, sizeof(data) - 1);
    close(fd);
    if(n <= 0){ return; }
    data[n] = '\0';
    char* line = data;
    while(*line){
        char* end = strchr(line, '\n');
        size_t len = end ? size_t(end - line) + 1 : strlen(line);
        for(int i = 0; i < nb_prefixes; i++){
            size_t prefix_len = strlen(prefixes[i]);
            if(len >= prefix_len && memcmp(line, prefixes[i], prefix_len) == 0){
                (void)!write(STDERR_FILENO, line, len);
                break;
            }
        }
        if(!end){ break; }
        line = end + 1;
    }
}

inline void handler(int sig, siginfo_t* info, void* uctx){
    const greg_t* regs = static_cast<ucontext_t*>(uctx)->uc_mcontext.gregs;
    char buf[512];
    int n = snprintf(buf, sizeof(buf), "\nFATAL signal %d code %d addr %p tid %d rip 0x%llx rsp 0x%llx",
        sig, info->si_code, info->si_addr, (int)gettid(),
        (unsigned long long)regs[REG_RIP], (unsigned long long)regs[REG_RSP]
    );
    if(__rseq_size > 0 && n > 0 && n < int(sizeof(buf))){
        const uint32_t* rs = reinterpret_cast<const uint32_t*>(
            static_cast<char*>(__builtin_thread_pointer()) + __rseq_offset
        );
        n += snprintf(buf + n, sizeof(buf) - n, " | rseq cpu_start %u cpu %u rseq_cs 0x%08x%08x flags %u",
            rs[0], rs[1], rs[3], rs[2], rs[4]
        );
    }
    if(n > 0 && n < int(sizeof(buf)) - 1){ buf[n++] = '\n'; }
    if(n > 0){ (void)!write(STDERR_FILENO, buf, size_t(n < int(sizeof(buf)) ? n : int(sizeof(buf)))); }

    const char* const meminfo[] = {"MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"};
    writeMatchingLines("/proc/meminfo", meminfo, 4);
    const char* const status[] = {"VmRSS:", "VmLck:", "VmPin:", "VmSwap:", "Threads:"};
    writeMatchingLines("/proc/self/status", status, 5);

    // SA_RESETHAND restored the default action: re-raise to get the core dump
    raise(sig);
}

} // namespace crash_diagnostics

inline void installCrashHandler(){
    struct sigaction sa = {};
    sa.sa_sigaction = crash_diagnostics::handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
}