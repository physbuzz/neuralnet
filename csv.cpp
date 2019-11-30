#include "csv.h"
#include <sstream>
#include <fstream>
#include <math.h>

bool loadRectCSV(std::string fname,RectCSV *data, int expected_w, int expected_h){
    std::ifstream datafile(fname.c_str());

    if(!datafile.is_open()){
        std::cerr<<"Error in loadRectCSV: couldn't open file "<<fname<<std::endl;
        return false;
    }
    data->w=expected_w; data->h=expected_h;
    data->data=std::vector<double>(expected_w*expected_h,0);

    bool success=loadRectCSVStream(datafile,data);

    datafile.close();
    return success;
}

/* Load data from the specified stream and populate the RectCSV* passed in.
 * Returns true on success, false otherwise. */
bool loadRectCSVStream(std::istream& st,RectCSV *data,int expected_w, int expected_h){
    //After maxindex we have to use push_back
    size_t maxindex=data->data.size();
    
    //current insertion index
    size_t index=0;

    //used to check width/height.
    int detectedWidth=0;
    int detectedHeight=0;

    //Begin parsing line by line
    std::string line;
    while((expected_h==0 || detectedHeight<expected_h) && std::getline(st,line)){
        std::istringstream iss(line);

        //detected width of the current line
        int linew=0;

        //Successful reading of characters
        bool success=true;
        while(success){
            double parsedDouble;
            char comma;
            success=bool(iss>>parsedDouble);
            
            //Read in the double or send an error flag.
            if(success) {
                if(index>=maxindex)
                    data->data.push_back(parsedDouble);
                else
                    data->data[index]=parsedDouble;
                index++;
                linew++;
            } else {
                std::cerr<<"Error reading double in loadCSVRectNonstrict."<<std::endl;
                return false;
            }
            //Only continue if we successfully read a comma, and we don't have eof or fail bits set.
            success=bool(iss>>comma);
            success=success && (comma==',');
            if(iss.fail() || iss.eof()) 
                break;
        }
        //Check tht the width is behaving as expected (rectangular file)
        if(detectedWidth>0){
            if(linew!=detectedWidth){
                std::cerr<<"Error in loadCSVRectNonstrict: nonrectangular csv detected. Width should be "<<detectedWidth<<" but is actually "<<linew<<std::endl;
                return false;
            }
        } else {
            detectedWidth=linew;
        }

        if(expected_w>0 && linew!=expected_w){
            std::cerr<<"Error in loadCSVRectNonstrict: detected width is "<<linew<<", but expected width is set to "<<expected_w<<"."<<std::endl;
            return false;
        }


        //Increase height
        detectedHeight++;
    }

    if(index<maxindex){
        std::cout<<"Warning in loadCSVRectNonstrict: More data allocated than values loaded!"<<std::endl;
    }

    data->w=detectedWidth;
    data->h=detectedHeight;

    return true;
}


bool saveRectCSV(std::string fname, const RectCSV& data){
    std::ofstream ofs(fname.c_str(),std::ofstream::out);
    if(ofs.is_open()){
        bool success = writeRectCSVStream(ofs,data);
        ofs.close();
        return success;
    }

    std::cerr<<"Error in saveRectCSV. Could not open file "<<fname<<std::endl;
    return false;
}
bool writeRectCSVStream(std::ostream& stream,const RectCSV& data) {
    for(int i=0;i<data.h;i++){
        for(int j=0;j<data.w;j++){
            stream<<data.data.at(i*data.w+j);
            if(j<data.w-1)
                stream<<",";
        }
        stream<<std::endl;
    }
    return true;
}

/*
RectCSV loadRectCSV(std::string fname){

    return 0;
}

    RectCSV ret={3,3,std::vector<double>(9,0)};
    return ret;
}
*/

bool saveWeightCSV(std::string fname, const WeightCSV& data){
    std::ofstream ofs(fname.c_str(),std::ofstream::out);
    if(ofs.is_open()){
        bool success = writeWeightCSVStream(ofs,data);
        ofs.close();
        return success;
    }

    std::cerr<<"Error in saveWeightCSV. Could not open file "<<fname<<std::endl;
    return false;
}
bool writeWeightCSVStream(std::ostream& stream, const WeightCSV& data){
    size_t L=data.n.size();
    if(L<2 || data.weights.size()!=L-1){
        std::cout<<L<<", "<<data.weights.size()<<std::endl;
        std::cerr<<"Error in writeWeightCSVStream. Invalid data (<2 neuron layers or weights.size() is not n.size()-1)"<<std::endl;
        return false;
    } 

    //Write n[i] line.
    for(size_t i=0;i<L;i++){
        stream<<data.n[i];
        if(i!=L-1)
            stream<<",";
    }
    stream<<std::endl;
    for(size_t l=0;l<L-1;l++){
        int nrows=data.n[l];
        int ncols=data.n[l+1];
        if(data.weights[l].size()!=nrows*ncols) {
            std::cerr<<"Error in writeWeightCSVStream. Invalid data (data.weights[l] is not of dimensions n[l] x n[l+1] for l="<<l<<")"<<std::endl;
            return false;
        } 

        for(size_t row=0;row<nrows;row++){
            for(size_t col=0;col<ncols;col++){
                stream<<data.weights[l][row*ncols+col];
                if(col!=ncols-1)
                    stream<<",";
            }
            stream<<std::endl;
        }
    }
    return true;
}

bool loadWeightCSV(std::string fname, WeightCSV* data){
    std::ifstream datafile(fname.c_str());

    if(!datafile.is_open()){
        std::cerr<<"Error in loadWeightCSV: couldn't open file "<<fname<<std::endl;
        return false;
    }

    bool success=loadWeightCSVStream(datafile,data);

    datafile.close();
    return success;
}

WeightCSV rectsToWeightCSV(const RectCSV& nr, const std::vector<RectCSV>& weights){
    WeightCSV ret;
    size_t L=nr.data.size();
    for(size_t l=0; l<L;l++){
        int nl=int(nr.data[l]); 
        ret.n.push_back(nl);
    }
    for(size_t l=0; l<L-1;l++){
        if(weights[l].w!=ret.n[l+1] || weights[l].h!=ret.n[l]) {
            std::cerr<<"error in rectsToWeightCSV: malformed input"<<std::endl;
        }
        ret.weights.push_back(weights[l].data);
    }
    return ret;
}

bool loadWeightCSVStream(std::istream& stream, WeightCSV* data){
    WeightCSV ret;

    bool success=true;

    //Load the first row: the number of neurons in each layer
    RectCSV nr;
    success=loadRectCSVStream(stream,&nr,0,1);
    if(!success)
        return false;

    //Convert the first row to integers
    size_t L=nr.data.size();
    for(size_t l=0; l<L;l++){
        int nl=int(nr.data[l]); 
        ret.n.push_back(nl);
    }

    for(size_t l=0;l<L-1;l++){
        RectCSV weight;
        success=loadRectCSVStream(stream,&weight,ret.n[l+1],ret.n[l]);
        if(!success)
            return false;
        ret.weights.push_back(weight.data);
    }
    *data=ret;

    return false;
}

