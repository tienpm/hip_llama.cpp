#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <vector>

#define MAX_NUM_BLOCK 4
#define MAX_EMBEDDING_LEN 1024
using namespace std;


int64_t generate_adress() {
  // Get the current time in milliseconds
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  auto epoch = now_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
  int64_t unique_code = value.count();

  return unique_code;
}

// struct BlockTableElement {
//     int64_t adress;
//     float * filled_block;
// };

// TLB: translation lookaside buffer
class TLB {
  /* stay on host , and its adress is logical adress
  * store adress of physical adress

  */

  // pair (key, value) với key là  adress và value là số block đã được lưu trữ trong 1  adress
private:
  std::vector<std::pair<int64_t, int>> block_table;

public:
  // add 1 block to 1 adress, kiểm tra xem adress đã tồn tại trong block_table chưa, nếu chưa thì
  // thêm vào block_table
  void add_adress(int64_t adress) {
    // nếu đã tồn tại,  tăng giá trị value lên 1 nếu thoả mãn điều kiện value < MAX_NUM_BLOCK
    if (block_table.size() > 0) {
      for (auto it = block_table.begin(); it != block_table.end(); it++) {
        if (it->first == adress) {
          if (it->second < MAX_NUM_BLOCK) {
            it->second++;
          } else {
            printf("Can't add block to adress %d\n", (int)adress);
          }
          return;
        }
      }
    }
  }

  // remove  adress from block_table
  void remove(int64_t adress) {
    for (auto it = block_table.begin(); it != block_table.end(); it++) {
      if (it->first == adress) {
        block_table.erase(it);
        return;
      }
    }
    printf("Can't find adress %d\n", (int)adress);
  }

  // lấy ra số block đã được lưu trữ trong 1  adress
  int get_num_block(int64_t adress) {
    if (block_table.size() > 0) {
      for (auto it = block_table.begin(); it != block_table.end(); it++) {
        if (it->first == adress) {
          return it->second;
        }
      }
    }
    printf("Can't find adress %d\n", (int)adress);
    return -1;
  }
};


class Logical_KV_Block {
  /* stay on host , and its adress is logical adress
   * store embedding of token in a request
   * is a array with e array has 4 elements (4 blocks)
   */

private:
  // tạo 1 vector, mỗi phần tử của vector này sau này sẽ lưu trữ 1 pair (key, value) với key là
  // logical adress và value là 1 vector chứa 4 block, mỗi block là 1 array dạng float
  int max_num_block = MAX_NUM_BLOCK;
  std::vector<std::pair<int64_t, std::vector<float *>>> logical_kv_block;

public:
  // thêm 1 block vào 1 logical adress bằng cách thêm và cuối của vector chứa max_num_block block và
  // cũng là cuối của vector logical_kv_block chứa pair (key, value)
  void add_block(int64_t logical_adress, float *block) {
    if (logical_kv_block.size() == 0
        || logical_kv_block[logical_kv_block.size() - 1].second.size() == max_num_block) {
      logical_kv_block.push_back(std::make_pair(logical_adress, std::vector<float *>()));
    }

    if (logical_kv_block.size() > 0
        && logical_kv_block[logical_kv_block.size() - 1].second.size() < max_num_block) {
      logical_kv_block[logical_kv_block.size() - 1].second.push_back(block);
    }
  }

  // lấy ra 4 block của 1 logical adress
  std::vector<float *> get_block(int64_t logical_adress) {
    for (auto it = logical_kv_block.begin(); it != logical_kv_block.end(); it++) {
      if (it->first == logical_adress) {
        return it->second;
      }
    }
    printf("Can't find logical adress %d\n", (int)logical_adress);
    return std::vector<float *>();
  }

  // lấy ra tất cả các logical adress và 4 block của nó sau đó ghép vào theo thứ tự để thành 1
  // vector
  std::vector<float *> get_all_block() {
    std::vector<float *> all_block;
    for (auto it = logical_kv_block.begin(); it != logical_kv_block.end(); it++) {
      for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
        all_block.push_back(*it2);
      }
    }
    return all_block;
  }

  // lấy ra logical adress đầu tiên
  int64_t get_first_logical_adress() { return logical_kv_block[0].first; }

  // lấy ra logical adress cuối cùng
  int64_t get_last_logical_adress() { return logical_kv_block[logical_kv_block.size() - 1].first; }

  // lấy ra tất cả các logical adress
  std::vector<int64_t> get_all_logical_adress() {
    std::vector<int64_t> all_logical_adress;
    for (auto it = logical_kv_block.begin(); it != logical_kv_block.end(); it++) {
      all_logical_adress.push_back(it->first);
    }
    return all_logical_adress;
  }

  // lấy ra tất cả các physical adress và 4 block của nó sau đó ghép vào theo thứ tự để thành 1
  // vector
  std::vector<float *> get_all_block_with_physical_adress() {
    std::vector<float *> all_block;
    for (auto it = logical_kv_block.begin(); it != logical_kv_block.end(); it++) {
      for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
        all_block.push_back(*it2);
      }
    }
    return all_block;
  }
};



class Physical_KV_Block {
  /* stay on gpu , and its adress is physical adress of gpu
   * store embedding of token in a request
   * is a array with e array has 4 elements (4 blocks)
   * Physical_KV_Block có giới hạn số vector có thể chứa, mỗi vector chứa 4 hoặc max_num_block
   * blockes
   */

private:
  // tạo 1 vector, mỗi phần tử của vector này sau này sẽ lưu trữ 1 pair (key, value) với key là
  // physical adress và value là 1 vector chứa 4 block, mỗi block là 1 array dạng float
  std::vector<std::pair<int64_t, std::vector<float *>>> physical_kv_block;
  int max_num_block = MAX_NUM_BLOCK;
  int max_num_vector;
  int num_current_vector = 0;

public:
  // set max_vector vectors có thể chứa
  void set_max_num_vector(int max_num_vector) { this->max_num_vector = max_num_vector; }

  // thêm 1 block vào 1 physical adress, nếu phần tử physical_kv_block chứa max_num_block blockes
  // thì thêm 1 pair mới vào physical_kv_block pair mới này sẽ chứa 1 vector mới chứa 4 blockes và
  // có 1 physical adress mới
  void add_block(int64_t physical_adress, float *block) {
    // nếu tổng số phần tử chưa đủ max_num_vector thì
    if (physical_kv_block.size() == 0
        || (physical_kv_block[physical_kv_block.size() - 1].second.size() == max_num_block
            && num_current_vector < max_num_vector)) {
      if (num_current_vector < max_num_vector) {
        physical_kv_block.push_back(std::make_pair(physical_adress, std::vector<float *>()));
        num_current_vector++;
      } else {
        // nếu phần tử cuối cùng của physical_kv_block chưa đủ max_num_block blockes thì thêm block
        // vào phần tử cuối cùng đó
        if (physical_kv_block.size() > 0
            && physical_kv_block[physical_kv_block.size() - 1].second.size() < max_num_block) {
          physical_kv_block[physical_kv_block.size() - 1].second.push_back(block);
        }
      }
    }

    // nếu num_current_vector đã đủ max_num_vector và phần tử cuối cùng của physical_kv_block đã đủ
    // max_num_block blockes thì xoá phần tử đầu tiên của physical_kv_block và thêm 1 pair mới vào
    // physical_kv_block
    else

    {
      if (num_current_vector == max_num_vector
          && physical_kv_block[physical_kv_block.size() - 1].second.size() == max_num_block) {
        physical_kv_block.erase(physical_kv_block.begin());
        physical_kv_block.push_back(std::make_pair(physical_adress, std::vector<float *>()));
      }
    }
  }

  // lấy ra 4 block của 1 physical adress
  std::vector<float *> get_block(int64_t physical_adress) {
    for (auto it = physical_kv_block.begin(); it != physical_kv_block.end(); it++) {
      if (it->first == physical_adress) {
        return it->second;
      }
    }
    printf("Can't find physical adress %d\n", (int)physical_adress);
    return std::vector<float *>();
  }

  // tìm sau đó xoá 1 pair (key, value) trong physical_kv_block
  void remove(int64_t physical_adress) {
    for (auto it = physical_kv_block.begin(); it != physical_kv_block.end(); it++) {
      if (it->first == physical_adress) {
        physical_kv_block.erase(it);
        num_current_vector--;
        return;
      }
    }
    printf("Can't find physical adress %d\n", (int)physical_adress);
  }
};

int main() {
    // create
    TLB block_table;
    Logical_KV_Block logical_kv_block;
    Physical_KV_Block physical_kv_block;

    int64_t physical_adress = generate_adress();
    // create a random block with bock_size = 1024 float elements
    float *block;
    block = (float *)malloc(1024 * sizeof(float));
    for (int i = 0; i < 1024; i++) {
        block[i] = i;
    }

    /*
    * add block
    */
    // add block to physical_kv_block
    logical_kv_block.add_block(physical_adress, block);

    //
    block_table.add_adress(physical_adress);
    physical_kv_block.add_block(physical_adress, block);

    // get block
    std::vector<float *> block1 = logical_kv_block.get_block(physical_adress);
    
    // print some elements of block1
    for (int i = 0; i < 10; i++) {
        printf("%f\n", block1[0][i]);
    }

  return 0;
}
