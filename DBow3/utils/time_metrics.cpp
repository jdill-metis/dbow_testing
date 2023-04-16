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

#include <boost/filesystem.hpp>

using namespace boost::filesystem;
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
    string eroc = "/home/ningjiang/Downloads/mav0/cam0/data";
    string av = "/home/ningjiang/Downloads/images";
    string def = "/home/ningjiang/metis/DBoW_testing/DBoW2/demo/images";
    string mixed = "/media/ningjiang/ADRIAN/temp/processed/mixed";
    string indoor = "/media/ningjiang/ADRIAN/temp/processed/indoor";
    string outdoor = "/media/ningjiang/ADRIAN/temp/processed/outdoor";

    path p(outdoor);
    // int j = 0;
    for (auto i = directory_iterator(p); i != directory_iterator(); i++)
    {
        if (!is_directory(i->path())) //we eliminate directories
        {
            // j++;
            // cout << i->path().filename().string() << endl;
            string local_path = outdoor+"/"+i->path().filename().string();
            cout << local_path << endl;
            paths.push_back(local_path);
        }
        else
            continue;
    }
    // cout<<"num images: " <<j << endl;

    // for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
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
        // cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw std::runtime_error("Could not open image "+path_to_images[i]);
        // cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        // cout<<"done detecting features"<<endl;
    }
    return features;
}

void testVocCreation(const vector<cv::Mat> &features)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 7;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;

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
    // cout << "Matching images against themselves (0 low, 1 high):" << endl;
    // cout << "Matching images against themselves (1st vs first 20): " << endl;
    // BowVector v1, v2;
    //     voc.transform(features[0], v1);

    // for(size_t i = 0; i < 20; i++)
    // {
    //     voc.transform(features[i], v2);
    //     double score1 = voc.score(v1, v2);
    //     cout << "Image " << i << ": " << score1 << endl;
    // //     for(size_t j = 0; j < features.size(); j++)
    // //     {
    // //         voc.transform(features[j], v2);

    // //         double image_score = voc.score(v1, v2);
    // //         cout << "Image " << i << " vs Image " << j << ": " << image_score << endl;
    // //     }
    // }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("dbow3_outdoor_k9l7.dbow3");
    cout << "Done" << endl;

}

void testVocLoad(const vector<cv::Mat> &features)
{
    cout << endl << "loading vocabulary..." << endl;

    const int k = 9;
    const int L = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;
    DBoW3::Vocabulary voc(k, L, weight, score);

    double av_load_time = 0;
    double av_transform_times[5] = {0};
    for (int f = 0; f < 1; f++)
    {
        auto t_start=std::chrono::high_resolution_clock::now();
        voc.load("orb_outdoor.dbow3");
        // voc.load("fbow_large_k9l5.fbow");
        // voc.save("testing_ORBvoc.yml.gz");
        auto t_end=std::chrono::high_resolution_clock::now();
        double load_time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
        cout<<"load time="<<load_time<<" ms"<<endl;
        av_load_time += load_time;

        int iterations =1000;
        //test transforms for iterations times
        for (int j = 0; j < 4; j++)
        {
            if (j == 1) iterations = 2000;
            if (j == 2) iterations = 4000;
            if (j == 3) iterations = 10000;
            if (j == 4) iterations = 20000;
            if (j == 5) iterations = 100000;

            BowVector vv, vv2;
            voc.transform(features[1],vv2);
            t_start= std::chrono::high_resolution_clock::now();
            for(int i=0;i<iterations;i++)
                voc.transform(features[0],vv);
            t_end= std::chrono::high_resolution_clock::now();
            double time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
            // cout<<"vocab transform time "<< iterations <<" iterations="<<time<<" ms"<<endl;
            av_transform_times[j]+= time;
            // t_start= std::chrono::high_resolution_clock::now();
            // for(int i=0;i<iterations;i++)
            //     voc.transform(features[0],vv);
            //     double image_score = voc.score(vv, vv2);
            // t_end= std::chrono::high_resolution_clock::now();
            // time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
            // cout<<"vocab transform + score calulation time " << iterations << " iterations="<<time<<" ms"<<endl;
        }
    }
    cout << "average vocab load time of testing_eroc_k10_L4=" << (av_load_time / 1) << " ms" << endl;
    int iterations = 1000;
    for (int j = 0; j < 4; j++)
    {
        if (j == 1)
            iterations = 2000;
        if (j == 2)
            iterations = 4000;
        if (j == 3)
            iterations = 10000;
        if (j == 4)
            iterations = 20000;
        if (j == 5)
            iterations = 100000;
        cout << "average vocab transform time " << iterations << " iterations=" << (av_transform_times[j] / 15) << " ms" << endl;
    }
    
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
        vector< cv::Mat   >   features= loadFeatures(images,descriptor);
        testVocCreation(features);

        // testVocLoad(features);

        // testDatabase(features);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
