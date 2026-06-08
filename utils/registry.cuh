#pragma once
#include <string>
#include <utility>
#include <vector>

using SgemmFn = void (*)(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C);

struct KernelEntry {
  std::string name;
  SgemmFn fn;
};

class KernelRegistry {
public:
  static KernelRegistry &instance() {
    static KernelRegistry reg;
    return reg;
  }
  void add(std::string name, SgemmFn fn) {
    entries_.push_back({std::move(name), fn});
  }
  const std::vector<KernelEntry> &entries() const { return entries_; }

private:
  KernelRegistry() = default;
  std::vector<KernelEntry> entries_;
};

struct KernelRegistrar {
  KernelRegistrar(std::string name, SgemmFn fn) {
    KernelRegistry::instance().add(std::move(name), fn);
  }
};

#define REGISTER_KERNEL(display_name)                                           \
  static KernelRegistrar registrar(display_name, launcher)
