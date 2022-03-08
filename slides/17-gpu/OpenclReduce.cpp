#include <Rcpp.h>
#include <boost/compute.hpp>

using namespace Rcpp;

namespace compute = boost::compute;

// [[Rcpp::export]]
float OpenclReduce(std::vector<float>& h_x) {
    
    // get the default compute device
    compute::device device = compute::system::default_device();
    
    // create a compute context and command queue
    compute::context context(device);
    compute::command_queue queue(context, device);
    
    // allocate memory on the device
    compute::vector<float> d_x(h_x.size(), context);
    
    // copy data from the host to the device
    compute::copy(h_x.begin(), h_x.end(), d_x.begin(), queue);
    
    // calculate the sum and save the result back to the host
    float total;
    compute::reduce(d_x.begin(), d_x.end(), &total, queue);
    
    return total;
}
