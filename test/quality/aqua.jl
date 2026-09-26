using Aqua
using MultiSymplectic
using Test

Aqua.test_all(MultiSymplectic;
    undefined_exports = (broken = true,),                                       # issue #3
    stale_deps = false,                                                         # issue #4: run as @test_broken below
    deps_compat = (broken = true, check_weakdeps = (broken = true,)),           # issue #5
    piracies = (broken = true,))                                                # issue #6

# `test_stale_deps` takes no `broken` keyword, so its check is run here directly.
@test_broken isempty(Aqua.find_stale_deps(Base.PkgId(MultiSymplectic)))  # issue #4
