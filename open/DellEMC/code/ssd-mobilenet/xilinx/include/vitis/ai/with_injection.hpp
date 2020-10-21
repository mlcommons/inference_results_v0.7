/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <dlfcn.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wundefined-var-template"
#endif
namespace vitis {
namespace ai {

template <typename T>
struct WithInjection {
 public:
  template <typename... Args>
  using factory_method_t =
      typename std::add_pointer<std::unique_ptr<T>(Args&&...)>::type;
  // make it easier to access this class, e.g. T::with_injection_t
  using with_injection_t = WithInjection<T>;
  template <typename... Args>
  static std::unique_ptr<T> create(Args&&... args) {
    if (the_factory_method<Args&&...> == nullptr) {
      std::abort();
    }
    auto ret = the_factory_method<Args&&...>(std::forward<Args>(args)...);
    dynamic_cast<WithInjection<T>*>(ret.get())->initialize();
    return ret;
  }

  struct so_name_t {
    so_name_t(const char* n) : so_name{n}, sym_name("factory_method") {}
    so_name_t(const char* n, const char* s) : so_name{n}, sym_name{s} {}
    const char* so_name;
    const char* sym_name;
  };
  template <typename... Args>
  static std::unique_ptr<T> create(so_name_t name, Args&&... args) {
    auto handle =
        dlopen((std::string("lib") + name.so_name + std::string(".so")).c_str(),
               RTLD_LAZY);
    if (!handle) {
      abort();
    }
    typename std::add_pointer<factory_method_t<Args...>>::type factory_method_p;
    factory_method_p = (decltype(factory_method_p))dlsym(handle, name.sym_name);
    if (factory_method_p == nullptr) {
      std::abort();
    }
    auto ret = (*factory_method_p)(std::forward<Args>(args)...);
    ret->initialize();
    return ret;
  }

 public:
  template <typename Subclass>
  struct factory_method_generator_t {
    template <typename... Args>
    static constexpr factory_method_t<Args&&...> generate() {
      return [](Args&&... args) -> std::unique_ptr<T> {
        return std::make_unique<Subclass>(std::forward<Args>(args)...);
      };
    }
  };

 private:
  virtual void initialize() {}

 private:
  template <typename... Args>
  static factory_method_t<Args&&...> the_factory_method;
};

#define DECLARE_INJECTION(super, imp, args...)                                 \
  namespace vitis {                                                            \
  namespace ai {                                                               \
  template <>                                                                  \
  template <>                                                                  \
  super::factory_method_t<args>                                                \
      super::with_injection_t::the_factory_method<args> =                      \
          super::factory_method_generator_t<imp>::generate<args>();            \
  }                                                                            \
  }

#define DECLARE_INJECTION_IN_SHARED_LIB_WITH_SYMBOL_NAME(super, imp, name,     \
                                                         args...)              \
  extern "C" super::factory_method_t<args> name;                               \
  super::factory_method_t<args> name =                                         \
      super::factory_method_generator_t<imp>::generate<args>();

#define DECLARE_INJECTION_IN_SHARED_LIB(super, imp, args...)                   \
  DECLARE_INJECTION_IN_SHARED_LIB_WITH_SYMBOL_NAME(super, imp, factory_method, \
                                                   args)

}  // namespace ai
}  // namespace vitis
