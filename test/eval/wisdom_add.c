// add FFTW wisdom of the DFT sizes N ... to file (forward and backward, the
// same planning flags as sdr_gen_fftw_wisdom()), e.g. the sizes of the 24 MHz
// evaluation with the decimation: 16000 48000 64000 160000 320000
// usage: wisdom_add file N ...   (build: gcc -O2 -o wisdom_add wisdom_add.c -lfftw3f)
// (for pocket_trk: -opt <conf> with the line "fftw_wisdom = <file>")
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
int main(int argc, char **argv)
{
    if (argc < 3) return 1;
    fftwf_import_wisdom_from_filename(argv[1]);
    for (int i = 2; i < argc; i++) {
        int N = atoi(argv[i]);
        fftwf_complex *a = fftwf_alloc_complex(N), *b = fftwf_alloc_complex(N);
#if defined(__aarch64__)
        unsigned flag = FFTW_MEASURE;
#else
        unsigned flag = FFTW_PATIENT;
#endif
        fftwf_plan p = fftwf_plan_dft_1d(N, a, b, FFTW_FORWARD, flag);
        fftwf_plan q = fftwf_plan_dft_1d(N, b, a, FFTW_BACKWARD, flag);
        fftwf_destroy_plan(p); fftwf_destroy_plan(q);
        fftwf_free(a); fftwf_free(b);
        printf("N=%d\n", N); fflush(stdout);
        fftwf_export_wisdom_to_filename(argv[1]);
    }
    return 0;
}
