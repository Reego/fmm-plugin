#include "bli_plugin_fmm_blis.h"


double TIMES[] = { 0.0, 0.0, 0.0, 0.0 };
int CLOCK_CALLS[] = { 0, 0, 0, 0 };

double _bl_clock()
{
    return _bl_clock_helper();
}

#if BL_OS_WINDOWS
// --- Begin Windows build definitions -----------------------------------------

double _bl_clock_helper()
{
    LARGE_INTEGER clock_freq = {0};
    LARGE_INTEGER clock_val;
    BOOL          r_val;

    r_val = QueryPerformanceFrequency( &clock_freq );

    if ( r_val == 0 )
    {
        fprintf( stderr, "\nblislab: %s (line %lu):\nblislab: %s \n", __FILE__, __LINE__, "QueryPerformanceFrequency() failed" );
        fflush( stderr );
        abort();
    }

    r_val = QueryPerformanceCounter( &clock_val );

    if ( r_val == 0 )
    {
        fprintf( stderr, "\nblislab: %s (line %lu):\nblislab: %s \n", __FILE__, __LINE__, "QueryPerformanceFrequency() failed" );
        fflush( stderr );
        abort();
    }

    return ( ( double) clock_val.QuadPart / ( double) clock_freq.QuadPart );
}

// --- End Windows build definitions -------------------------------------------
#elif BL_OS_OSX
// --- Begin OSX build definitions -------------------------------------------

double _bl_clock_helper()
{
    mach_timebase_info_data_t timebase;
    mach_timebase_info( &timebase );

    uint64_t nsec = mach_absolute_time();

    double the_time = (double) nsec * 1.0e-9 * timebase.numer / timebase.denom;

    if ( _gtod_ref_time_sec == 0.0 )
        _gtod_ref_time_sec = the_time;

    return the_time - _gtod_ref_time_sec;
}

// --- End OSX build definitions ---------------------------------------------
#else
// --- Begin Linux build definitions -------------------------------------------

double _bl_clock_helper()
{
    double the_time, norm_sec;
    struct timespec ts;

    clock_gettime( CLOCK_MONOTONIC, &ts );

    if ( _gtod_ref_time_sec == 0.0 )
        _gtod_ref_time_sec = ( double ) ts.tv_sec;

    norm_sec = ( double ) ts.tv_sec - _gtod_ref_time_sec;

    the_time = norm_sec + ts.tv_nsec * 1.0e-9;

    return the_time;
}

// --- End Linux build definitions ---------------------------------------------
#endif