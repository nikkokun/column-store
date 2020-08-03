#include <iostream>
#include <string>
#include <map>
#include <math.h>
#include <tuple>
#include <fstream>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <sys/stat.h>
#include <algorithm>

using namespace std;

class BigVectorInterface {
 public:
  virtual bool flush_to_disk(string split_id) = 0;
  virtual int get_size() = 0;
};

class MemoryManager {
 private:
  int max_size;
  int current_size;

  map<string, BigVectorInterface *> split_map_; //<split, bigvector>
  deque<string> dq; //split

 public:
  MemoryManager(int capacity) {
    split_map_.clear();
    dq.clear();
    MemoryManager::max_size = capacity;
    MemoryManager::current_size = 0;
  }

  bool get(string split_id) {
    // not found
    if (!split_map_.count(split_id)) {
      return false;
    } else {
      // found
      deque<string>::iterator it = dq.begin();
      while (*it!=split_id) {
        it++;
      }
      // update queue: update it to most recent used value
      dq.erase(it);
      dq.push_front(split_id);
      return true;
    }
  }

  void put(string split_id, BigVectorInterface *bigvector, int size) {
    // not present in cache
    if (!split_map_.count(split_id)) {
      // check if cache is full
      while ((size + MemoryManager::current_size) > MemoryManager::max_size) {
        cout << "flushing victim to disk..\n";
        string lru_split = dq.back();
        int split_to_free_size = split_map_[lru_split]->get_size();
        split_map_[lru_split]->flush_to_disk(lru_split);
        MemoryManager::current_size -= split_to_free_size;
        dq.pop_back();
        split_map_.erase(lru_split);
      }

    } else {
      // present in cache, remove it from queue and map
      deque<string>::iterator it = dq.begin();
      while (*it!=split_id) {
        it++;
      }

      dq.erase(it);
      split_map_.erase(split_id);
    }

    // update the cache
    dq.push_front(split_id);
    split_map_[split_id] = bigvector;
    MemoryManager::current_size += size;
  }

};

template<typename T>
class BigVector : public BigVectorInterface {
 private:
  //variables
  int col_id_;
  int rows;
  int lines_per_split_;
  string storage_dir_;
  MemoryManager *memory_manager;
  //mappings
  map<string, int> id_split_map_; //<uuid, split>
  map<int, string> split_id_map_; //<uuid, split>
  map<int, vector<T> *> memory_storage_;
  map<int, bool> loaded_splits_; //split -> bool is loaded mapping
  map<int, tuple<int, int>> index_map_; //<absolute index, (split, relative_index)> mapping

  //private methods
  string get_split_fp(int split) {
    return storage_dir_ + "/" + to_string(split) + ".tsv";
  }

 public:

  BigVector(const int col_id, const int lines_per_split, const string &storage_dir, MemoryManager *memory_manager) {
    BigVector::col_id_ = col_id;
    BigVector::lines_per_split_ = lines_per_split;
    BigVector::storage_dir_ = storage_dir + "/" + "col_" + std::to_string(col_id);
    BigVector::memory_manager = memory_manager;
    BigVector::rows = 0;

    mkdir(BigVector::storage_dir_.c_str(), 0777);
  }

  BigVector(const int rows, const int col_id, const int lines_per_split, const string &storage_dir, MemoryManager *memory_manager) {
    BigVector::lines_per_split_ = lines_per_split;
    auto uuid = boost::uuids::random_generator()();
    BigVector::storage_dir_ = storage_dir + "/" + "vector_" +  boost::uuids::to_string(uuid);
    BigVector::memory_manager = memory_manager;
    BigVector::rows = 0;

    mkdir(BigVector::storage_dir_.c_str(), 0777);
    T default_val;
    for(int i = 0; i < rows; ++i) {
      BigVector::append_to_disk(to_string(default_val), i);
    }
  }

  void append_to_disk(string element, int index) {
    int relative_index = index%BigVector::lines_per_split_;
    int split = floor((float) index/(float) BigVector::lines_per_split_);
    BigVector::index_map_[index] = make_pair(split, relative_index);

    string fp = get_split_fp(split);
    ofstream outfile;
    outfile.open(fp, ios_base::app);
    outfile << element << "\n";
    outfile.close();
    ++BigVector::rows;
  }

  bool load_memory(int split) {
    // assign split id if it doesn't exist
    if (!split_id_map_.count(split)) {
      string uuid = boost::uuids::to_string(boost::uuids::random_generator()());
      split_id_map_[split] = uuid;
      id_split_map_[uuid] = split;
    }

    if (loaded_splits_.count(split)) {
      if (loaded_splits_[split]) {
        return true;
      }
    }
    string fp = get_split_fp(split);
    ifstream input(fp);

    if (input) {
      //call memory manager put operation
      BigVector::memory_manager->put(split_id_map_[split], this, BigVector::lines_per_split_);
      vector<T> *loaded_vector;
      if (memory_storage_.count(split)) {
        loaded_vector = memory_storage_[split];
      } else {
        loaded_vector = new vector<T>;
      }
      T value;
      while (input >> value) {
        loaded_vector->push_back(value);
      }
      memory_storage_[split] = loaded_vector;
      loaded_splits_[split] = true;
      return true;
    } else {
      return false;
    }
  }

  virtual bool flush_to_disk(string split_id) {
    if (!id_split_map_.count(split_id)) {
      cout << "Split ID not found, exiting\n";
      exit(0);
    }
    int split = BigVector::id_split_map_[split_id];
    if (loaded_splits_.count(split)) {
      if (!loaded_splits_[split]) {
        cout << loaded_splits_[split];
        return false;
      }
    }
    string fp = get_split_fp(split);
    ofstream outfile(fp, ios::trunc);
    copy(memory_storage_[split]->begin(), memory_storage_[split]->end(), std::ostream_iterator<T>(outfile, "\n"));
    outfile.close();
    memory_storage_[split]->clear();
    loaded_splits_[split] = false;
    return loaded_splits_[split];
  }

  virtual int get_size() {
    return BigVector::lines_per_split_;
  }

  T& operator[](int i) {

    if (i > BigVector::rows) {
      cout << "Index out of bound, exiting";
      exit(0);
    }

    tuple<int, int> split_index = index_map_[i];
    int split = get<0>(split_index);
    int relative_index = get<1>(split_index);
    if (loaded_splits_.count(split)) {
      if (loaded_splits_[split]) {
        //call memory manager get operation
        BigVector::memory_manager->get(split_id_map_[split]);
        return memory_storage_[split]->at(relative_index);
      } else {
        load_memory(split);
        return memory_storage_[split]->at(relative_index);
      }
    }
    load_memory(split);
    return memory_storage_[split]->at(relative_index);

  }

};

template<typename T>
class BigMatrix {

 private:
  int cols_ = 0;
  string storage_dir_;
  map<int, BigVector<T> *> column_map_;
  MemoryManager *memory_manager;

  vector<string> split(string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find(delimiter, pos_start))!=string::npos) {
      token = s.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + delim_len;
      res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
  }

 public:

  BigMatrix(const string &input_fp,
            const string &delimiter,
            const int lines_per_split,
            const string &data_dir,
            MemoryManager *memory_manager) {
    auto uuid = boost::uuids::random_generator()();
    BigMatrix::storage_dir_ = data_dir + "matrix_" + boost::uuids::to_string(uuid) + "/";
    mkdir(BigMatrix::storage_dir_.c_str(), 0777);
    BigMatrix::memory_manager = memory_manager;

    int idx = 0;
    ifstream input(input_fp);
    if (input.is_open()) {
      string line;
      while (getline(input, line)) {
        auto row = split(line, delimiter);
        for (size_t i = 0; i < row.size(); ++i)
          if (idx==0) {
            ++BigMatrix::cols_;
            column_map_.insert(pair<int, BigVector<T> *>(i,
                                                         new BigVector<T>(i,
                                                                          lines_per_split,
                                                                          BigMatrix::storage_dir_,
                                                                          BigMatrix::memory_manager)));
            column_map_[i]->append_to_disk(row[i], idx);
          } else {
            column_map_[i]->append_to_disk(row[i], idx);
          }
        ++idx;
      }
    }
  }

  BigMatrix(const int rows,
            const int cols,
            const string &delimiter,
            const int lines_per_split,
            const string &data_dir,
            MemoryManager *memory_manager) {

    auto uuid = boost::uuids::random_generator()();
    BigMatrix::storage_dir_ = data_dir + "matrix_" + boost::uuids::to_string(uuid) + "/";
    mkdir(BigMatrix::storage_dir_.c_str(), 0777);
    BigMatrix::memory_manager = memory_manager;

    vector<T> data(cols);
    for(int i = 0; i < rows; ++i) {
      for(size_t j = 0; j < data.size(); ++j) {
        if (i==0) {
          ++BigMatrix::cols_;
          column_map_.insert(pair<int, BigVector<T> *>(j, new BigVector<T>(j,lines_per_split, BigMatrix::storage_dir_, BigMatrix::memory_manager)));
          column_map_[j]->append_to_disk(to_string(data[j]), i);
        } else {
          column_map_[j]->append_to_disk(to_string(data[j]), i);
        }
      }
    }
  }

  BigVector<T>& operator[](int j) {
    if ((j) > BigMatrix::cols_) {
      cout << "Index out of bound, exiting";
      exit(0);
    }
    return *BigMatrix::column_map_[j];
  }

};

void normalize(BigMatrix<double>* f, int rows,int cols){
  int i,j;
  double f_temp;

  for(i=0;i<cols;i++){
    f_temp=0;
    for(j=0;j<rows;j++){
      f_temp += (*f)[i][j];
    }
    f_temp/=rows;
    for(j=0;j<rows;j++){
      (*f)[i][j]-=f_temp;
    }
    f_temp=0;
    for(j=0;j<rows;j++){
      f_temp+=(*f)[i][j]*(*f)[i][j];
    }
    f_temp/=rows;
    if(f_temp!=0){
      f_temp=sqrt(f_temp);
      for(j=0;j<rows;j++){
        (*f)[i][j] /= f_temp;
      }
    }
  }
}

void comp_conv(BigMatrix<double>* S, BigMatrix<double>* W, BigMatrix<double>* F, int n, int p, double rho){
  int i,j,k;
  double f_temp;

  for(i=0;i<p;i++){
    for(j=i;j<p;j++){
      f_temp=0;
      for(k=0;k<n;k++){
        f_temp+=((*F)[i][k]* (*F)[j][k]);
      }
      f_temp/=n;
      (*S)[j][i] = f_temp;
      (*S)[i][j] = (*S)[j][i];
      (*W)[j][i] = f_temp;
      (*W)[i][j]=(*W)[j][i];
    }
  }

  for(i=0;i<p;i++){
    (*W)[i][i] += rho;
  }
}

void copy_W(BigMatrix<double>* W, BigMatrix<double>* W_b,int p){
  int i,j;

  for(i=0;i<p;i++){
    for(j=0;j<p;j++){
      (*W_b)[i][j]= (*W)[i][j];
    }
  }
}

void update_W(BigMatrix<double>* S,
    BigMatrix<double>* W,
    int* beta_num,
    BigVector<int>* beta_p,
    BigVector<int>* beta_flag,
    BigVector<double>* beta_v,
    int p,
    double rho,
    int n,
    double* diff,
    BigVector<double>* beta,
    double epsilon
    ) {

  int i,j,u;
  double z, delta;

  for(i=0;i<p;i++){
    (*beta)[i]=0;
    (*beta_flag)[i]=0;
  }

  for(i=0;i<*beta_num;i++){
    u=(*beta_p)[i];
    (*beta_flag)[u]=1;
    (*beta)[u]=(*beta_v)[i];
  }

  do {

    delta=0;

    for(i=0;i<p;i++){
      if(i!=n){
        z=0;
        for(j=0;j<*beta_num;j++){
          u=(*beta_p)[j];
          if(u!=i){
            z+=(*beta)[u] * (*W)[i][u];
          }
        }

        z=(*S)[n][i]-z;

        if(z>rho){
          z-=rho;
        }
        else if(z <- rho){
          z+=rho;
        }
        else{
          z=0;
        }

        z /= (*W)[i][i];
        delta+= fabs((*beta)[i]-z);
        (*beta)[i]=z;

        if((*beta)[i] != 0 && (*beta_flag)[i] == 0){
          (*beta_p)[*beta_num]=i;
          *beta_num = *beta_num+1;
          (*beta_flag)[i]=1;
        }

        if((*beta)[i] == 0 && (*beta_flag)[i] == 1){
          (*beta_flag)[i]=0;
          for(j = 0; j < *beta_num; j++){
            u=(*beta_p)[j];
            if(u==i){
              (*beta_p)[j]=(*beta_p)[*beta_num-1];
              *beta_num=*beta_num-1;
              break;
            }
          }
        }
      }
    }
    delta/=p;
  } while(delta > epsilon);

  for(i = 0; i < *beta_num; i++){
    u=(*beta_p)[i];
    (*beta_v)[i]=(*beta)[u];
  }

  for(i=0;i<p;i++){
    if(i!=n){
      z=0;
      for(j=0;j<*beta_num;j++){
        u=(*beta_p)[j];
        z+=(*beta_v)[j]*(*W)[i][u];
      }
      (*W)[n][i]=z;
      (*W)[i][n]=(*W)[n][i];
    }
  }

  for(i=0;i<*beta_num;i++){
    u=(*beta_p)[i];
    (*beta_flag)[u]=0;
  }
}

double comp_diff(BigMatrix<double>* W, BigMatrix<double>* W_b,int p){
  int i,j;
  double diff=0;

  for(i=0;i<p;i++){
    for(j=i+1;j<p;j++){
      diff+=fabs((*W_b)[i][j]-(*W)[i][j]);
    }
  }
  diff/=(p*p-p);

  return(diff);
}

void estimate_inv(BigMatrix<double>* W, BigMatrix<double>* E, BigVector<int> *beta_num, BigMatrix<int>* beta_p, BigMatrix<double>* beta_v,int p){
  int i,j,u;
  double t;

  for(i=0;i<p;i++){
    for(j=0;j<p;j++){
      (*E)[i][j]=0;
    }
  }

  for(i=0;i<p;i++){
    t=0;
    for(j=0;j<(*beta_num)[i];j++){
      u=(*beta_p)[i][j];
      t+=(*beta_v)[i][j] * (*W)[i][u];
    }
    t=(*W)[i][i]-t;
    t=1/t;

    (*E)[i][i]=t;
    for(j = 0; j < (*beta_num)[i]; j++){
      u=(*beta_p)[i][j];
      (*E)[i][u] =- (*beta_v)[i][j]*t;
      (*E)[u][i] = (*E)[i][u];
    }
  }
}

void comp_diag(BigMatrix<double>* W, BigVector<int>* beta_num, BigMatrix<int>* beta_p, BigMatrix<double>* beta_v, int p, BigVector<double>* diag){
  int i,j,u;
  double t;

  for(i=0;i<p;i++){
    t=0;
    for(j = 0; j < (*beta_num)[i]; j++){
      u=(*beta_p)[i][j];
      t+=(*beta_v)[i][j]*(*W)[i][u];
    }
    t=(*W)[i][i]-t;
        t=1/t;
    (*diag)[i]=t;
  }
}



int main() {

  const int n = 10;
  const int p = 20;
  const float rho = 0.5;
  double diff = 0.0f;
  double epsilon=0.001;

  MemoryManager *memory_manager = new MemoryManager(10000);

  BigMatrix<double>* F = new BigMatrix<double>("/Users/nicoroble/graphical-lasso/column-store/data/test.tsv",
                                                  "\t",
                                                  5,
                                                  "/Users/nicoroble/graphical-lasso/column-store/data/",
                                                  memory_manager);
  BigMatrix<double>* W = new BigMatrix<double>(p, p, "\t", 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigMatrix<double>* W_b = new BigMatrix<double>(p, p, "\t", 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigMatrix<double>* S = new BigMatrix<double>(p, p, "\t", 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigMatrix<double>* E = new BigMatrix<double>(p, p, "\t", 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigMatrix<int>* beta_p = new BigMatrix<int>(p, p, "\t", 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigMatrix<double>* beta_v = new BigMatrix<double>(p, p, "\t", 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);

  BigVector<int>* beta_num = new BigVector<int>(p, 0, 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigVector<int>* beta_flag = new BigVector<int>(p, 0, 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigVector<double>* beta = new BigVector<double>(p, 0, 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);
  BigVector<double>* diag = new BigVector<double>(p, 0, 5, "/Users/nicoroble/graphical-lasso/column-store/data/", memory_manager);

  normalize(F, n, p);
  comp_conv(S,W,F,n,p,rho);

  for(int i = 0; i < p; i++){
    (*beta_num)[i]=0;
  }

  do {
    copy_W(W,W_b,p);

    for(int i = 0; i < p;i++){
      update_W(S, W, &(*beta_num)[i], &(*beta_p)[i], beta_flag, &(*beta_v)[i], p, rho, n, &diff, beta, epsilon);
    }
    diff=comp_diff(W,W_b,p);
  } while(diff>epsilon);

  estimate_inv(W,E,beta_num,beta_p,beta_v,p);
  comp_diag(W,beta_num,beta_p,beta_v,p,diag);

  return 0;
}