# Third-party code: the rules that make a project's copies of other people's
# code from their upstream clones. Included by IntrSharedPtr/Makefile and
# SharedPtr/Makefile; each has an `imports` target that makes its OWN
# directory's copy with the canned recipe below, and every build target there
# depends on that copy, so a plain `make` makes it first.
#
# Why: ThirdParty/<name> (the git clones at the top of the work tree) is the one
# source of third-party code we depend on; what the build uses is derived from
# it by these rules and a patch of our changes kept next to the output. Neither
# the clones nor the outputs are tracked by the repository (both are in
# .gitignore); this file, the Makefiles and the patches are.
#
#   IntrSharedPtr/mm_hp/         Maged Michael's hazard pointers
#                                (github.com/magedm/mm_hp), used by
#                                intr_shared_ptr_hp and hp_drain.h; clone
#                                HazardPtr, patch IntrSharedPtr/mm_hp.patch.
#   SharedPtr/lock_free_shared_ptr/parlay/
#                                Daniel Anderson's parlay::shared_ptr and hazard
#                                pointers (github.com/DanielLiamAnderson/atomic_shared_ptr),
#                                which our lock_free_shared_ptr/atomic_shared_ptr.hpp is
#                                built on, plus ParlayLib's headers for its pool allocator
#                                (github.com/cmuparlay/parlaylib), unmodified; clones
#                                AtomicSharedPtr and ParlayLib, patch
#                                SharedPtr/lock_free_shared_ptr/parlay.patch (folly
#                                replaced by standard equivalents, the list's Harris mark
#                                stripped in parlay's reference counting).
# Each patch's header says what each change is for. A project that only USES
# another project's copy (SharedPtr and LockFreeList use IntrSharedPtr/mm_hp/,
# LockFreeList uses SharedPtr's parlay/) never makes it itself: its rule for the
# copy's SOURCE.txt runs `$(MAKE) -C <that project> imports`. What mm_hp/ is to
# a build (its source, headers and stamp) is written once, for its owner and
# both consumers, in IntrSharedPtr/mm_hp.mk.
#
# When an output is remade. Its SOURCE.txt, written when it is made, is the
# stamp the build targets depend on; the stamp depends on the patch and on this
# file (which holds the pins and what is taken from each clone). Not on the
# generated files' own dates: git archive gives them upstream's commit dates,
# older than anything here. So editing a patch, a pin or this file remakes the
# output at the next build, and everything built with it is rebuilt (its targets
# list the stamp). Not on the including Makefile either, which changes often for
# unrelated reasons: after changing a stamp rule itself (the clones its
# `third_party_import` call names, or a different patch file, whose date may be
# older than the stamp), run `make -B imports`. `make -B` remakes the outputs
# too (it passes -B down to the sibling makes as well).
#
# The fleet. A build host may have the sources and the generated outputs made
# elsewhere (uploaded) but no clones; copy tools do not always preserve dates,
# so an uploaded patch may well be newer than the uploaded stamp. When NONE of
# an output's clones exists under THIRD_PARTY but the output does (it has its
# SOURCE.txt), the recipe warns and keeps it, unchecked: its SOURCE.txt says
# what it was made from. Without the output as well it fails. As soon as any of
# the clones exists, the output is remade from the clones, with every check
# below, and a missing or wrong clone fails the build: with clones present a
# stale output is never silently kept. (Failing in the fleet case instead
# would make a fresh upload unbuildable whenever the copy reordered dates;
# keeping the output whenever a clone check fails would hide a stale output on
# the machine that has the clones.)
#
# Linux only (GNU tar and patch, md5sum); the recipe runs under make's /bin/sh.

# This file as the includer names it (../third_party.mk): a prerequisite of the
# stamps. Taken first, while this file is still the last word of MAKEFILE_LIST.
THIRD_PARTY_MK := $(lastword $(MAKEFILE_LIST))

# Where the clones live: HazardPtr, AtomicSharedPtr and ParlayLib under the top
# of the work tree's ThirdParty/. `?=`, so that an environment variable or
# `make THIRD_PARTY=...` can point elsewhere (it reaches the sibling makes too).
# Absolute, because SOURCE.txt records it.
THIRD_PARTY ?= $(abspath $(dir $(THIRD_PARTY_MK))ThirdParty)

# Every clone an output is made from, by its directory name under THIRD_PARTY:
#   <clone>_COMMIT  the upstream commit the patch was made against; the clone
#                   must be checked out at it (another commit fails: the patch
#                   may not fit it; refresh the patch, then the pin), and the
#                   content is taken from it, never from the clone's working
#                   tree, so local edits there cannot leak into the output;
#   <clone>_URL     upstream, recorded in SOURCE.txt;
#   <clone>_PATHS   what is taken: paths inside the commit (empty: every file);
#   <clone>_STRIP   leading path components dropped on extraction.
#
# HazardPtr: every file (sources, README, licenses).
HazardPtr_COMMIT = b26e5edc779bf2afaea821549c04bf0a07b20f11
HazardPtr_URL = https://github.com/magedm/mm_hp
HazardPtr_PATHS =
HazardPtr_STRIP = 0
# ParlayLib: all of include/parlay/ (parlay/alloc.h pulls 12 more of its
# headers; copying the whole directory needs no list), include/parlay/<file>
# landing at <file>.
ParlayLib_COMMIT = 51017699dcc421f80479cdb238d3092233ad0d26
ParlayLib_URL = https://github.com/cmuparlay/parlaylib
ParlayLib_PATHS = include/parlay
ParlayLib_STRIP = 2
# AtomicSharedPtr: the four upstream files the adapter needs -- not
# basic_atomic_shared_ptr.hpp (needs folly's hazptr), fast_shared_ptr.hpp, or
# upstream's atomic_shared_ptr.hpp (our adapter replaces it) -- landing beside
# ParlayLib's headers. Extracted after ParlayLib (the order of the clones in
# the import call).
AtomicSharedPtr_COMMIT = 3c213ef8a437b419171f901e640ddfdd6056d905
AtomicSharedPtr_URL = https://github.com/DanielLiamAnderson/atomic_shared_ptr
AtomicSharedPtr_PATHS = $(addprefix include/parlay/,shared_ptr.hpp details/atomic_details.hpp \
                        details/hazard_pointers.hpp details/wait_free_counter.hpp)
AtomicSharedPtr_STRIP = 2

# $(call third_party_check_clone,<clone>): a shell fragment (one line, ending in
# `;`) that fails the recipe unless THIRD_PARTY/<clone> is the top of its own
# git clone (not a plain directory inside some other repository) checked out at
# <clone>_COMMIT. A clone with no pin is a make error.
third_party_check_clone = \
  $(if $($1_COMMIT),,$(error $(THIRD_PARTY_MK): no $1_COMMIT for the clone $1)) \
  dir='$(THIRD_PARTY)/$1'; \
  top=$$(git -C "$$dir" rev-parse --show-toplevel 2>/dev/null) || top=; \
  [ -n "$$top" ] && [ "$$(cd "$$top" && pwd -P)" = "$$(cd "$$dir" && pwd -P)" ] \
    || { echo "FAILED: $$dir is not a git clone (see $(notdir $(CURDIR))/README.md)" >&2; exit 1; }; \
  got=$$(git -C "$$dir" rev-parse HEAD); \
  [ "$$got" = '$($1_COMMIT)' ] \
    || { echo "FAILED: $$dir is at $$got; the patch was made against $($1_COMMIT)" >&2; exit 1; };

# $(call third_party_extract,<clone>): a shell fragment (one line, ending in `;`)
# that adds <clone>_PATHS of the clone at <clone>_COMMIT to the temporary
# directory $tmp and appends the clone's line for SOURCE.txt to $tmp.src. The
# archive goes through a file, not a pipe: /bin/sh has no pipefail, and a
# failing git archive must fail the recipe rather than feed tar nothing.
# $tmp.tar and $tmp.src are beside $tmp, not in it (they must not land in the
# output), and the recipe's cleanup removes them.
third_party_extract = \
  git -C '$(THIRD_PARTY)/$1' archive --format=tar $($1_COMMIT) $($1_PATHS) > "$$tmp.tar"; \
  tar -x -f "$$tmp.tar" --strip-components=$($1_STRIP) -C "$$tmp"; \
  rm -f "$$tmp.tar"; \
  echo '$1 $(or $(strip $($1_PATHS)),(every file)): $(THIRD_PARTY)/$1 at $($1_COMMIT) ($($1_URL))' >> "$$tmp.src";

# $(call third_party_import,<clones>): the recipe of an output's stamp rule,
#   <output dir>/SOURCE.txt: <patch> $(THIRD_PARTY_MK)
# The output directory ($(@D), relative to the including Makefile) and the
# patch ($<, the first prerequisite, applied inside the output with -p1) come
# from the rule, so each is spelt once; a target not named SOURCE.txt is a make
# error. <clones> are the clones the output is made from, in extraction order
# (later ones overwrite earlier ones' files of the same name). The patch is not
# checked for existence: as a prerequisite, make fails before the recipe runs.
# Flow, as one shell (so that a failure anywhere stops it and the cleanup trap
# covers every step):
#   1. The fleet case (header above): no clone exists and the output does --
#      warn, keep it, done. No clone and no output -- fail.
#   2. Every clone checked (third_party_check_clone) before anything is made,
#      so that a run is all or nothing.
#   3. A temporary directory beside the output (mktemp's 0700 widened to 755,
#      like a source directory) gets each clone's files (third_party_extract).
#   4. The patch, without fuzz: a hunk whose context does not match exactly
#      fails instead of landing nearby.
#   5. SOURCE.txt, written last (as an upstream file of that name could
#      otherwise overwrite it): who made it and when, each clone's line, the
#      patch and its md5.
#   6. The swap: the old output is renamed aside to $tmp.old, the new one moved
#      in, the old one removed. Only between the two renames (two system calls
#      apart) is no output in place; a run interrupted there gets the old one
#      put back by the cleanup, so an interruption never leaves the build
#      without one (removing the old output before moving the new one in could).
#      Another run of the same import at the same time (two top-level makes that
#      both reach `make -C ../IntrSharedPtr imports`) may win either rename. A
#      failed rename aside is therefore not an error by itself (the other run
#      moved the old output away first); the move in decides: if anything is in
#      place by then, our mv -T fails (-T: rather than moving our directory
#      INSIDE it), and the run still succeeds when what is in place has exactly
#      the content this run made (SOURCE.txt aside, which differs only in its
#      date) -- the other run's output, or an old output that could not be
#      renamed but needs no change. Anything else fails the run.
# Cleanup (EXIT trap; the signal trap turns an interrupt into an exit, so that
# it runs under dash as well): this run's own $tmp, $tmp.tar, $tmp.src and
# $tmp.old, nothing else -- a broader pattern would delete the files of another
# run of the same import going on at the same time (two top-level makes that
# both reach `make -C ../IntrSharedPtr imports`). A run killed outright
# (SIGKILL) leaves its <output dir>.tmp.* behind: .gitignore covers them, and
# they can be removed by hand while no import runs.
define third_party_import
@$(if $(strip $1),,$(error $(THIRD_PARTY_MK): third_party_import for $@ names no clones)) \
$(if $(filter SOURCE.txt,$(notdir $@)),,$(error $(THIRD_PARTY_MK): third_party_import must make <output dir>/SOURCE.txt, not $@)) \
set -eu; \
out='$(@D)'; patchfile='$<'; \
have_clone=; \
$(foreach c,$1,[ ! -e '$(THIRD_PARTY)/$c' ] || have_clone=1;) \
if [ -z "$$have_clone" ]; then \
  if [ -f "$$out/SOURCE.txt" ]; then \
    echo "WARNING: no clone of $(strip $1) under $(THIRD_PARTY): keeping $(notdir $(CURDIR))/$$out as made elsewhere (see its SOURCE.txt), unchecked against $$patchfile" >&2; \
    exit 0; \
  fi; \
  echo "FAILED: no $(notdir $(CURDIR))/$$out and no clone of $(strip $1) under $(THIRD_PARTY) to make it from (see $(notdir $(CURDIR))/README.md)" >&2; \
  exit 1; \
fi; \
$(foreach c,$1,$(call third_party_check_clone,$c)) \
tmp=; \
trap 'if [ -n "$$tmp" ]; then [ -e "$$out" ] || [ ! -e "$$tmp.old" ] || mv -T "$$tmp.old" "$$out"; rm -rf "$$tmp" "$$tmp.tar" "$$tmp.src" "$$tmp.old"; fi' EXIT; \
trap 'exit 1' HUP INT TERM; \
tmp=$$(mktemp -d "$$out.tmp.XXXXXX"); \
chmod 755 "$$tmp"; \
$(foreach c,$1,$(call third_party_extract,$c)) \
patch -d "$$tmp" -p1 --fuzz=0 --no-backup-if-mismatch --quiet < "$$patchfile"; \
{ \
  echo "Generated by make imports in $(notdir $(CURDIR))/ ($(THIRD_PARTY_MK)) on $$(date '+%Y-%m-%d %H:%M:%S %z'); do not edit."; \
  cat "$$tmp.src"; \
  echo "patched with $$(basename "$$patchfile"), md5 $$(md5sum < "$$patchfile" | cut -d' ' -f1)"; \
} > "$$tmp/SOURCE.txt"; \
if [ -e "$$out" ]; then mv -T "$$out" "$$tmp.old" 2>/dev/null || true; fi; \
if mv -T "$$tmp" "$$out" 2>/dev/null; then \
  echo "made $(notdir $(CURDIR))/$$out"; \
elif diff -r -q -x SOURCE.txt "$$tmp" "$$out" > /dev/null 2>&1; then \
  echo "made $(notdir $(CURDIR))/$$out (another run moved in the same content first)"; \
else \
  echo "FAILED: could not move $$tmp to $(notdir $(CURDIR))/$$out" >&2; exit 1; \
fi; \
rm -rf "$$tmp.old"
endef
