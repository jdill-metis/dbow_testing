// #include "dbow2/TemplatedVocabulary.h"
// #include "dbow2/FORB.h"
#include <opencv2/core/core.hpp>
#include "fbow.h"
#include <chrono>
#include "dbow3/DBoW3.h"

#include <set>
using namespace std;
// using ORBVocabulary=DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ;

namespace fbow{
 class VocabularyCreator{
public:
    //DBoW2

    //  struct ninfo{
    //      ninfo(){}
    //      ninfo(uint32_t Block,ORBVocabulary::Node *Node):block(Block),node(Node){}
    //      int64_t block=-1;
    //      ORBVocabulary::Node *node=0;
    //  };


    // static void convert(ORBVocabulary &voc,fbow::Vocabulary &out_voc){
    //     uint32_t nonLeafNodes=0;
    //     std::map<uint32_t,ninfo> nodeid_info;
    //     uint32_t LeafNodes=0;
        
    //     for(int i=0;i<voc.m_nodes.size();i++){
    //         auto &node=voc.m_nodes[i];
        
    //         if(!node.isLeaf()) nodeid_info.insert(std::make_pair(node.id,ninfo(nonLeafNodes++,&node)));
    //         else {
    //         nodeid_info.insert(std::make_pair(node.id,ninfo(-1,&node)));
    //         LeafNodes++;
    //         }
    //     }

    //     out_voc.setParams(8,voc.m_k,CV_8UC1,32,nonLeafNodes,"orb");
    //     cerr<<"creating     size="<<out_voc._params._total_size/(1024*1024)<<"Mb "<<out_voc._params._total_size<<" bytes"<<endl;

    //     for(int i=0;i<voc.m_nodes.size();i++){
    //         ORBVocabulary::Node &node= voc.m_nodes[i];

    //         if(!node.isLeaf()) {
    //             auto &n_info=nodeid_info[node.id];

    //             fbow::Vocabulary::Block binfo=out_voc.getBlock(n_info.block);
    //             binfo.setN(node.children.size());
    //             binfo.setParentId(node.id);
    //             bool areAllChildrenLeaf=true;
    //             std::sort(node.children.begin(),node.children.end());

    //             for(int c=0;c<node.children.size();c++){

    //                 auto &child_info=nodeid_info[node.children[c]];

    //                 binfo.setFeature(c, child_info.node->descriptor);
    //                 if (child_info.node->isLeaf())
    //                     binfo.getBlockNodeInfo(c)->setLeaf(child_info.node->word_id,child_info.node->weight);
    //                 else {
    //                     areAllChildrenLeaf=false;
    //                     binfo.getBlockNodeInfo(c)->setNonLeaf(child_info.block);
    //                 }
    //             }
    //             binfo.setLeaf(areAllChildrenLeaf);
    //         }
    //     }
    // }

    //DBoW3
    struct ninfo{
         ninfo(){}
         ninfo(uint32_t Block,DBoW3::Vocabulary::Node *Node):block(Block),node(Node){}
         int64_t block=-1;
         DBoW3::Vocabulary::Node *node=0;
     };


    static void convert(DBoW3::Vocabulary &voc,fbow::Vocabulary &out_voc){
        uint32_t nonLeafNodes=0;
        std::map<uint32_t,ninfo> nodeid_info;
        uint32_t LeafNodes=0;
        
        for(int i=0;i<voc.m_nodes.size();i++){
            auto &node=voc.m_nodes[i];
        
            if(!node.isLeaf()) nodeid_info.insert(std::make_pair(node.id,ninfo(nonLeafNodes++,&node)));
            else {
            nodeid_info.insert(std::make_pair(node.id,ninfo(-1,&node)));
            LeafNodes++;
            }
        }

        out_voc.setParams(8,voc.m_k,CV_8UC1,32,nonLeafNodes,"orb");
        cerr<<"creating     size="<<out_voc._params._total_size/(1024*1024)<<"Mb "<<out_voc._params._total_size<<" bytes"<<endl;

        for(int i=0;i<voc.m_nodes.size();i++){
            DBoW3::Vocabulary::Node &node= voc.m_nodes[i];

            if(!node.isLeaf()) {
                auto &n_info=nodeid_info[node.id];

                fbow::Vocabulary::Block binfo=out_voc.getBlock(n_info.block);
                binfo.setN(node.children.size());
                binfo.setParentId(node.id);
                bool areAllChildrenLeaf=true;
                std::sort(node.children.begin(),node.children.end());

                for(int c=0;c<node.children.size();c++){

                    auto &child_info=nodeid_info[node.children[c]];

                    binfo.setFeature(c, child_info.node->descriptor);
                    if (child_info.node->isLeaf())
                        binfo.getBlockNodeInfo(c)->setLeaf(child_info.node->word_id,child_info.node->weight);
                    else {
                        areAllChildrenLeaf=false;
                        binfo.getBlockNodeInfo(c)->setNonLeaf(child_info.block);
                    }
                }
                binfo.setLeaf(areAllChildrenLeaf);
            }
        }
    }
 };
}

int main(int argc,char **argv){
    // if (argc!=3){cerr<<"Usage voc.txt out.fbow"<<endl;return -1;}

    //DBoW2

    // ORBVocabulary voc;
    // cout<<"loading dbow2 voc"<< endl;
    // // argv[1]
    // voc.load("ORBvoc.txt");
    // cout<<"done"<<endl;
    // double av_conv_time = 0;
    // fbow::Vocabulary fvoc;
    // for (int j = 0; j < 15; j++)
    // {
    //     auto t_start=std::chrono::high_resolution_clock::now();
    //     fbow::VocabularyCreator::convert(voc,fvoc);
    //     auto t_end=std::chrono::high_resolution_clock::now();

    //     double time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
    //     av_conv_time+=time;
    //     // cout<<"conversion time="<<time<<" ms"<< endl;
    // }
    // cout<<"av conversion time="<<(av_conv_time/15)<<" ms"<< endl;
    // //argv[2]
    // fvoc.saveToFile("testing_ORBvoc.fbow");

    //DBoW3

    DBoW3::Vocabulary voc;
    cout<<"loading dbow3 voc"<< endl;
    // argv[1]
    // voc.load("ORBvoc.txt");
    std::string s1 = "orb_outdoor.dbow3";
    std::string s2 = ".dbow3";
    std::string s3 = ".txt";

    if (s1.find(s2) != std::string::npos) {

    }
    voc.load("orb_outdoor.dbow3");
    cout<<"done"<<endl;
    double av_conv_time = 0;
    fbow::Vocabulary fvoc;
    for (int j = 0; j < 1; j++)
    {
        auto t_start=std::chrono::high_resolution_clock::now();
        fbow::VocabularyCreator::convert(voc,fvoc);
        auto t_end=std::chrono::high_resolution_clock::now();

        double time = double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
        av_conv_time+=time;
        // cout<<"conversion time="<<time<<" ms"<< endl;
    }
    cout<<"av conversion time="<<(av_conv_time/15)<<" ms"<< endl;
    //argv[2]
    fvoc.saveToFile("testing_ORBvoc_DBoW3.fbow");


}