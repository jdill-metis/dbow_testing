#include "dbow2/TemplatedVocabulary.h"
#include "dbow2/FORB.h"
#include <vector>


// DBoW3
#include "dbow3/DBoW3.h"
#include "dbow3/DescManip.h"
#include "dbow3/BowVector.h"


#include "fbow.h"
#include <chrono>
#include <opencv2/flann.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

using namespace std;

vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
vector<cv::Mat> loadFeatures(vector<string> path_to_images,string descriptor);
void testVocCreation(const vector<vector<cv::Mat > > &features);

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// number of training images
const int NIMAGES = 4;

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


vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

vector< cv::Mat  >  loadFeatures(vector<string> path_to_images,string descriptor="") {
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")   fdetector=cv::ORB::create(2000);

    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,  0,  3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(15, 4, 2 );
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
        if(image.empty())throw runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout<<"done detecting features"<<endl;
    }
    return features;
}

void testVocCreation(const vector<cv::Mat > &features)
{
    const int k = 9;
    const int L = 3;
    // const DBoW3::WeightingType weight = DBoW3::TF_IDF;
    // const DBoW3::ScoringType score = DBoW3::L1_NORM;
}

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<=2){
            cerr<<"Usage:  descriptor_name     image0 image1 ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
             return -1;
        }

        string descriptor=argv[1];

        auto images=readImagePaths(argc,argv,2);
        vector<cv::Mat> features = loadFeatures(images,descriptor);
        testVocCreation(features);


        // testDatabase(features);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }
}