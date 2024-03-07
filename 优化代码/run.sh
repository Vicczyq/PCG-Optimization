set -x

bsub -b -I -q q_sw_cpc2023 -swrunarg '-p' -perf -o out.log -shared -n 1 -cgsp 64 -host_stack 1024 -share_size 1500 -ldm_share_mode 5 -ldm_share_size 32 -cache_size 32 ./pcg_solve
