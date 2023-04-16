//First step of creating a vocabulary is extracting features from a set of images. We save them to a file for next step
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "fbow.h"

#include "vocabulary_creator.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "dirreader.h"

#include <boost/filesystem.hpp>

using namespace boost::filesystem;
using namespace fbow;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};

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
    for (auto i = directory_iterator(p); i != directory_iterator(); i++)
    {
        if (!is_directory(i->path())) //we eliminate directories
        {
            // cout << i->path().filename().string() << endl;
            string local_path = outdoor+"/"+i->path().filename().string();
            cout << local_path << endl;
            paths.push_back(local_path);
        }
        else
            continue;
    }

    // for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
    return paths;
}


auto ToDescriptorVector(const cv::Mat &descriptors)->std::vector<cv::Mat> {
    std::vector<cv::Mat> v_desc;
    v_desc.reserve(descriptors.rows);
    for (int j=0;j<descriptors.rows;j++)
        v_desc.push_back(descriptors.row(j));

    return v_desc;
}

auto ToDescriptorMatrix(std::vector<cv::Mat> v_descriptor)->cv::Mat {
    cv::Mat desc;
    // int rows = v_descriptor.size();
    int rows = v_descriptor[0].size().height;
    desc = v_descriptor[0];
    for (int j=1;j<rows;j++)
        cv::hconcat(desc, v_descriptor[j], desc);
    return desc;
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

    int j = 0;
    cout << "Extracting   features..." << endl;
    // for(size_t i = 0; i < path_to_images.size(); ++i)
    for(size_t i = 0; i < path_to_images.size(); ++i)

    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        // cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        // cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        // descriptors = ToDescriptorMatrix(ToDescriptorVector(descriptors) );
        features.push_back(ToDescriptorMatrix(ToDescriptorVector(descriptors) ));
        // cout<<"done detecting features"<<endl;
        j++;
    }
    std::cout << "num features: " << j << std::endl;
    std::cout << "num features: " << path_to_images.size() << std::endl;

    return features;
}

void testVocCreation(const vector<cv::Mat> &features, string desc_name="")
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 7;
    // const WeightingType weight = TF_IDF;
    // const ScoringType score = L1_NORM;

    fbow::VocabularyCreator::Params params;
    params.k = k;
    params.L = L;
    params.nthreads = 1;
    params.maxIters = 1;
    srand(0);

    fbow::VocabularyCreator voc_creator;
    fbow::Vocabulary voc;
    cout << "Creating a " << params.k << "^" << params.L << " vocabulary..." << endl;
    auto t_start=std::chrono::high_resolution_clock::now();
    voc_creator.create(voc,features,desc_name, params);
    auto t_end=std::chrono::high_resolution_clock::now();
    cout << "... done!" << endl;
    cout<<"time="<<double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count())<<" msecs"<<endl;
    cout<<"nblocks="<<voc.size()<<endl;
    cout<<"desktype="<<voc.getDescType()<<endl;

    // DBoW3::Vocabulary voc(k, L, weight, score);

    // lets do something with this vocabulary
    // cout << "Matching images against themselves (0 low, 1 high): " << endl;
    // BowVector v1, v2;
    // for(size_t i = 0; i < features.size(); i++)
    // {
    //     voc.transform(features[i], v1);
    //     for(size_t j = 0; j < features.size(); j++)
    //     {
    //         voc.transform(features[j], v2);

    //         double score = voc.score(v1, v2);
    //         cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    //     }
    // }
    // vector<map<double, int> > scores;


    cout << "Matching images against themselves (1st vs first 20): " << endl;
    fbow::fBow vv, vv2;
    vv = voc.transform(features[2]);
    // int avgScore = 0;
    // int counter = 0;
    // t_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i<4; i++)
    {
        // size_t j = i*30;
        vv2 = voc.transform(features[i]);
        double score1 = vv.score(vv, vv2);
        cout << "Image " << i << ": " << score1 << endl;


    //     map<double, int> score;
    //     for (size_t j = 0; j<features.size(); ++j)
    //     {

    //         vv2 = voc.transform(features[j]);
    //         double score1 = vv.score(vv, vv2);
    //         counter++;
    //         //		if(score1 > 0.01f)
    //         {
    //             score.insert(pair<double, int>(score1, j));
    //         }
    //         printf("%f, ", score1);

    //         cout << "Image " << i << " vs Image " << j << ": " << score1 << endl;
    //     }
    //     printf("\n");
    //     scores.push_back(score);

    }

    // t_end = std::chrono::high_resolution_clock::now();
    // avgScore += double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
    // std::cout << "avg score: " << avgScore << " # of features: " << features.size() << std::endl;


    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.saveToFile("outdoor_k9l7.fbow");
    cout << "Done" << endl;
}

void testVocLoad(const vector<cv::Mat> &features)
{
    // fbow::Vocabulary voc;
    // cout<<"loading fbow voc...."<<endl;
    // auto t_start=std::chrono::high_resolution_clock::now();
    // voc.readFromFile("fbow_small_voc.txt");
    // auto t_end=std::chrono::high_resolution_clock::now();
    // double fbow_load=double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    // cout<<"load time="<<fbow_load<<" ms"<<endl;

    double av_load_time = 0;
    double av_transform_times[5] = {0};

    for (int f = 0; f < 1; f++){

        fbow::Vocabulary voc;
        cout<<"loading fbow voc...."<<endl;
        auto t_start=std::chrono::high_resolution_clock::now();
        // voc.readFromFile("testing_ORBvoc.fbow");
        voc.readFromFile("testing_ORBvoc_DBoW3.fbow");
        cout << endl << "testing_ORBvoc_DBoW3.fbow" << endl;
        
        auto t_end=std::chrono::high_resolution_clock::now();
        double fbow_load=double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
        // cout<<"load time="<<fbow_load<<" ms"<<endl;
        av_load_time+=fbow_load;
        // cout << "Matching images against themselves (1st vs first 20): " << endl;
        // fbow::fBow vv, vv2;

        // cout <<"feature type1 : " << features[0].type() << endl;
        // cout <<"feature type2 : " << features[1].type() << endl;

        // vv = voc.transform(features[0]);

        // for (size_t i = 0; i<30; ++i)
        // {
        //     size_t j = i*30; //i*30;
        //     vv2 = voc.transform(features[j]);
        //     double score1 = vv.score(vv, vv2);
        //     cout << "Image " << i << ": " << score1 << endl;
        // }

        int iterations =1000;
        //test transforms for iterations times
        for (int j = 0; j < 4; j++)
        {
            if (j == 1) iterations = 2000;
            if (j == 2) iterations = 4000;
            if (j == 3) iterations = 10000;
            if (j == 4) iterations = 20000;
            if (j == 5) iterations = 100000;

            fBow vv, vv2;
            vv2 = voc.transform(features[1]);
            t_start= std::chrono::high_resolution_clock::now();
            for(int i=0;i<iterations;i++)
                vv == voc.transform(features[0]);
            t_end= std::chrono::high_resolution_clock::now();
            double time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
            // cout<<"vocab transform time "<< iterations <<" iterations="<<time<<" ms"<<endl;
            av_transform_times[j]+=time;
            // t_start= std::chrono::high_resolution_clock::now();
            // for(int i=0;i<iterations;i++)
            //     vv = voc.transform(features[0]);
            //     double image_score = vv.score(vv, vv2);
            // t_end= std::chrono::high_resolution_clock::now();
            // time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
            // cout<<"vocab transform + score calulation time " << iterations << " iterations="<<time<<" ms"<<endl;
        }
    }
    cout << "average vocab load time of testing_eroc_k9_L3=" << (av_load_time / 15) << " ms " << endl;
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

void testDatabase(const  vector<cv::Mat > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    // Vocabulary voc("small_voc.yml.gz");
    fbow::Vocabulary voc;
    voc.readFromFile("small_voc.yml.gz");

//     Database db(voc, false, 0); // false = do not use direct index
//     // (so ignore the last param)
//     // The direct index is useful if we want to retrieve the features that
//     // belong to some vocabulary node.
//     // db creates a copy of the vocabulary, we may get rid of "voc" now

//     // add images to the database
//     for(size_t i = 0; i < features.size(); i++)
//         db.add(features[i]);

//     cout << "... done!" << endl;

//     cout << "Database information: " << endl << db << endl;

//     // and query the database
//     cout << "Querying the database: " << endl;

//     QueryResults ret;
//     for(size_t i = 0; i < features.size(); i++)
//     {
//         db.query(features[i], ret, 4);

//         // ret[0] is always the same image in this case, because we added it to the
//         // database. ret[1] is the second best match.

//         cout << "Searching for Image " << i << ". " << ret << endl;
//     }

//     cout << endl;

//     // we can save the database. The created file includes the vocabulary
//     // and the entries added
//     cout << "Saving database..." << endl;
//     db.save("small_db.yml.gz");
//     cout << "... done!" << endl;

//     // once saved, we can load it again
//     cout << "Retrieving database once again..." << endl;
//     Database db2("small_db.yml.gz");
//     cout << "... done! This is: " << endl << db2 << endl;
}


// ----------------------------------------------------------------------------

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

        testVocCreation(features, descriptor);
        // testVocLoad(features);

        // testDatabase(features);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}