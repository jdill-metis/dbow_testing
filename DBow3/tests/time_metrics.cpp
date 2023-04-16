#include <iostream>
#include <chrono>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;


void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

vector<string> readImagePaths(int argc,char **argv,int start){
    vector<string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
        return paths;
}

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout<<"done detecting features"<<endl;
    }
    return features;
}

void testVocCreation(const vector<cv::Mat> &features)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;

    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;

    auto t_start=std::chrono::high_resolution_clock::now();
    voc.create(features);
    auto t_end=std::chrono::high_resolution_clock::now();

    cout << "... done!" << endl;
    double time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    cout<<"vocab creation time="<<time<<" ms"<<endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;
    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high):" << endl;
    BowVector v1, v2;
    t_start=std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < features.size(); i++)
    {
        voc.transform(features[i], v1);
        for(size_t j = 0; j < features.size(); j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;


    //test transforms for 1000 times
    BowVector vv, vv2;
    voc.transform(features[1],vv2);
    t_start= std::chrono::high_resolution_clock::now();
    for(int i=0;i<1000;i++)
        voc.transform(features[0],vv);
    t_end= std::chrono::high_resolution_clock::now();
    time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    cout<<"vocab transform time 1000 iterations="<<time<<" ms"<<endl;

    t_start= std::chrono::high_resolution_clock::now();
    for(int i=0;i<1000;i++)
        voc.transform(features[0],vv);
        double score = voc.score(vv, vv2);
    t_end= std::chrono::high_resolution_clock::now();
    time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    cout<<"vocab transform + score calulation time 1000 iterations="<<time<<" ms"<<endl;
}

void testVocLoad()
{
    const int k = 9;
    const int L = 5;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    DBoW3::Vocabulary voc(k, L, weight, score);

    auto t_start=std::chrono::high_resolution_clock::now();
    // voc.load("small_voc.yml.gz");
    voc.load("fbow_large_k9l5.fbow");
    auto t_end=std::chrono::high_resolution_clock::now();
    double load_time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    cout<<"load time="<<load_time<<" ms"<<endl;
}