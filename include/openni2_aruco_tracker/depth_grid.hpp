#include <opencv2/opencv.hpp>

class DepthGrid
{
public:
    DepthGrid();
    
private:

    cv::Size grid_size;
    const float depth_dec;
}
