/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <chrono>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <boost/filesystem.hpp>


using namespace boost::filesystem;
using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
//3682
 int NIMAGES = 0;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

vector<string> readImagePaths(){
    vector<string> paths;
    string eroc = "/home/ningjiang/Downloads/mav0/cam0/data";
    string av = "/home/ningjiang/Downloads/semseg11";
    string def = "/home/ningjiang/metis/DBoW_testing/DBoW2/demo/images";
    path p(def);
    // int j = 0;
    for (auto i = directory_iterator(p); i != directory_iterator(); i++)
    {
        if (!is_directory(i->path())) //we eliminate directories
        {
            // cout << i->path().filename().string() << endl;

            string local_path = def +"/"+ i->path().filename().string();
            // cout << local_path << endl;
            paths.push_back(local_path);
            // j++;
            // cout << j << endl;
            NIMAGES++;
        }
        else
            continue;
    }
    cout << "the number of images" << NIMAGES << endl;

    // for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
    return paths;
}

void loadFeatures(vector<vector<cv::Mat > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  int j =0;
  cout << "Extracting ORB features..." << endl;
  vector<string> paths = readImagePaths();
  for(int i = 0; i < paths.size(); ++i)
  {
    j++;
    // stringstream ss;
    // ss << "images/image" << i << ".png";

    cv::Mat image = cv::imread(paths[i], 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
    std::cout << "num features: " << j << std::endl;

}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 15;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L2_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  auto t_start=std::chrono::high_resolution_clock::now();
  voc.load("testing_orb_vocab.txt");
  // voc.create(features);
  auto t_end= std::chrono::high_resolution_clock::now();
  double time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());

  cout << "... done! creation time is: " << time << "ms" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // // lets do something with this vocabulary
  // cout << "Matching images against themselves (0 low, 1 high): " << endl;
    cout << "Matching images against themselves (1st vs first 20): " << endl;
  BowVector v1, v2;
        voc.transform(features[0], v1);

  
  for(int i = 0; i < 30; i++)
  {
    size_t j = i*30;
        voc.transform(features[j], v2);
        double score1 = voc.score(v1, v2);
        cout << "Image " << i << ": " << score1 << endl;
  //   for(int j = 0; j < NIMAGES; j++)
  //   {
  //     voc.transform(features[j], v2);
      
  //     double score = voc.score(v1, v2);
  //     cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.saveToTextFile("testing_eroc.txt");
  voc.save("testing_eroc_k15_L6.yml.gz");
  cout << "Done" << endl;
}

void testVocMetrics(const vector<vector<cv::Mat > > &features)
{
  double av_load_time = 0;
  double av_transform_times[5] = {};

  for (int f = 0; f < 1; f++) {
    const int k = 12;
    const int L = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L2_NORM;

    OrbVocabulary voc(k, L, weight, scoring);

    auto t_start=std::chrono::high_resolution_clock::now();
    voc.load("ORBvoc.zip");
    voc.save("testing_orb_vocab.yml.gz");
    
    // auto t_end=std::chrono::high_resolution_clock::now();
    // double time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    // // cout<<"vocab load time of testing_eroc_k10_L5="<<time<<" ms for "<< NIMAGES<< " images"<<endl;
    // av_load_time += time;
    // int iterations =1000;
    // //test transforms for iterations times
    // for (int j = 0; j < 4; j++)
    // {
    //     if (j == 1) iterations = 2000;
    //     if (j == 2) iterations = 4000;
    //     if (j == 3) iterations = 10000;
    //     if (j == 4) iterations = 20000;
    //     if (j == 5) iterations = 100000;

    //     BowVector vv, vv2;
    //     voc.transform(features[1],vv2);
    //     t_start= std::chrono::high_resolution_clock::now();
    //     for(int i=0;i<iterations;i++)
    //         voc.transform(features[0],vv);
    //     t_end= std::chrono::high_resolution_clock::now();
    //     time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    //     // cout<<"vocab transform time "<< iterations <<" iterations="<<time<<" ms"<<endl;

    //     av_transform_times[j]+=time;

    //     // t_start= std::chrono::high_resolution_clock::now();
    //     // for(int i=0;i<iterations;i++)
    //     //     voc.transform(features[0],vv);
    //     //     double image_score = voc.score(vv, vv2);
    //     // t_end= std::chrono::high_resolution_clock::now();
    //     // time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    //     // cout<<"vocab transform + score calulation time " << iterations << " iterations="<<time<<" ms"<<endl;
    // }
  }
  // cout << "average vocab load time of testing_eroc_k15_L6=" << (av_load_time / 15) << " ms for " << NIMAGES << " images" << endl;
  // int iterations = 1000;
  // for (int j = 0; j < 4; j++)
  // {
  //   if (j == 1)
  //       iterations = 2000;
  //   if (j == 2)
  //       iterations = 4000;
  //   if (j == 3)
  //       iterations = 10000;
  //   if (j == 4)
  //       iterations = 20000;
  //   if (j == 5)
  //       iterations = 100000;
  //   cout << "average vocab transform time " << iterations << " iterations=" << (av_transform_times[j] / 15) << " ms" << endl;
  // }

}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------

int main()
{
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  // testVocCreation(features);

  testVocMetrics(features);
  // wait();

  // testDatabase(features);

  return 0;
}