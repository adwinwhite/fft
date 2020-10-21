#ifndef FFT_UNIPROCESSOR_H
#define FFT_UNIPROCESSOR_H


namespace myfft {
    typedef double mcomplex[2];

    mcomplex* cfft_c(mcomplex* samples, const unsigned& order, const bool isInverse=false);

    mcomplex* cfft(mcomplex* samples, const unsigned& order, const bool isInverse=false);
}


#endif // FFT_UNIPROCESSOR_H
