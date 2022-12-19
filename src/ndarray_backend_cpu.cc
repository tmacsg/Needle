#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <malloc.h>

#ifdef _WIN32
#define WINDOWS 1
#elif
#define WINDOWS 0
#endif

namespace needle
{
  namespace cpu
  {

#define ALIGNMENT 256
#define TILE 8
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    /**
     * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
     * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
     * here by default.
     */
    struct AlignedArray
    {
      AlignedArray(const size_t size)
      {
        if (WINDOWS)
        {
          ptr = (scalar_t *)malloc(size * ELEM_SIZE);

          // _aligned_malloc not worked!!!
          // void* ptr_ = _aligned_malloc(size * ELEM_SIZE, ALIGNMENT);
          // if (ptr_ == NULL) throw std::bad_alloc();
          // memset(ptr_, 0, size * ELEM_SIZE);
          // ptr = (scalar_t*)ptr_;
        }
        else
        {
          // int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
          // if (ret != 0) throw std::bad_alloc();
        }

        this->size = size;
      }
      ~AlignedArray() { free(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }
      scalar_t *ptr;
      size_t size;
    };

    void Fill(AlignedArray *out, scalar_t val)
    {
      /**
       * Fill the values of an aligned array with val
       */
      for (int i = 0; i < out->size; i++)
      {
        out->ptr[i] = val;
      }
    }

    void recursiveLoops(std::vector<std::vector<int32_t>> &ret,
                        std::vector<int32_t> &indexes,
                        std::vector<int32_t> &end_indexes,
                        size_t pos)
    {
      if (pos == indexes.size())
      {
        std::vector<int32_t> temp;
        for (size_t i = 0; i < pos; i++)
          temp.push_back(indexes[i]);
        ret.push_back(temp);
      }
      else
      {
        for (indexes[pos] = 0; indexes[pos] != end_indexes[pos]; indexes[pos]++)
        {
          recursiveLoops(ret, indexes, end_indexes, pos + 1);
        }
      }
    }

    void Compact(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,
                 std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory
       *
       * Args:
       *   a: non-compact represntation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       *
       * Returns:
       *  void (you need to modify out directly, rather than returning anything; this is true for all the
       *  function will implement here, so we won't repeat this note.)
       */
      /// BEGIN YOUR SOLUTION
      std::vector<std::vector<int32_t>> permutations;
      std::vector<int32_t> indexes(shape.size(), 0);

      recursiveLoops(permutations, indexes, shape, 0);

      size_t cnt = 0;
      for (size_t i = 0; i < permutations.size(); i++)
      {
        size_t index = 0;
        for (size_t j = 0; j < shape.size(); j++)
        {
          index += strides[j] * permutations[i][j];
        }
        out->ptr[cnt++] = a.ptr[index + offset];
      }
      /// END YOUR SOLUTION
    }

    void EwiseSetitem(const AlignedArray &a, AlignedArray *out, std::vector<int32_t> shape,
                      std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items in a (non-compact) array
       *
       * Args:
       *   a: _compact_ array whose items will be written to out
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *out* array (not a, which has compact strides)
       *   offset: offset of the *out* array (not a, which has zero offset, being compact)
       */
      /// BEGIN YOUR SOLUTION
      std::vector<std::vector<int32_t>> permutations;
      std::vector<int32_t> indexes(shape.size(), 0);

      recursiveLoops(permutations, indexes, shape, 0);

      std::vector<int32_t> compact_strides(shape.size(), 1);
      for (int i = shape.size() - 2; i >= 0; i--)
      {
        compact_strides[i] = compact_strides[i + 1] * shape[i + 1];
        std::cout << compact_strides[i] << " ";
      }

      for (size_t i = 0; i < permutations.size(); i++)
      {
        size_t index = 0;
        size_t compact_index = 0;
        for (size_t j = 0; j < shape.size(); j++)
        {
          index += strides[j] * permutations[i][j];
          compact_index += compact_strides[j] * permutations[i][j];
        }
        out->ptr[index + offset] = a.ptr[compact_index];
      }
      /// END YOUR SOLUTION
    }

    void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out, std::vector<int32_t> shape,
                       std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items is a (non-compact) array
       *
       * Args:
       *   size: number of elements to write in out array (note that this will note be the same as
       *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
       *         product of items in shape, but covenient to just pass it here.
       *   val: scalar value to write to
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension of out
       *   strides: strides of the out array
       *   offset: offset of the out array
       */

      /// BEGIN YOUR SOLUTION
      std::vector<std::vector<int32_t>> permutations;
      std::vector<int32_t> indexes(shape.size(), 0);

      recursiveLoops(permutations, indexes, shape, 0);

      for (size_t i = 0; i < permutations.size(); i++)
      {
        size_t index = 0;
        for (size_t j = 0; j < shape.size(); j++)
        {
          index += strides[j] * permutations[i][j];
        }
        out->ptr[index + offset] = val;
      }
      /// END YOUR SOLUTION
    }

    void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      /**
       * Set entries in out to be the sum of correspondings entires in a and b.
       */
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] + b.ptr[i];
      }
    }

    void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      /**
       * Set entries in out to be the sum of correspondings entry in a plus the scalar val.
       */
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] + val;
      }
    }

    /**
     * In the code the follows, use the above template to create analogous elementise
     * and and scalar operators for the following functions.  See the numpy backend for
     * examples of how they should work.
     *   - EwiseMul, ScalarMul
     *   - EwiseDiv, ScalarDiv
     *   - ScalarPower
     *   - EwiseMaximum, ScalarMaximum
     *   - EwiseEq, ScalarEq
     *   - EwiseGe, ScalarGe
     *   - EwiseLog
     *   - EwiseExp
     *   - EwiseTanh
     *
     * If you implement all these naively, there will be a lot of repeated code, so
     * you are welcome (but not required), to use macros or templates to define these
     * functions (however you want to do so, as long as the functions match the proper)
     * signatures above.
     */

    /// BEGIN YOUR SOLUTION
    void EwiseMul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] * b.ptr[i];
      }
    }

    void ScalarMul(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] * val;
      }
    }

    void EwiseDiv(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] / b.ptr[i];
      }
    }

    void ScalarDiv(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] / val;
      }
    }

    void ScalarPower(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = pow(a.ptr[i], val);
      }
    }

    void EwiseMaximum(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] > b.ptr[i] ? a.ptr[i] : b.ptr[i];
      }
    }

    void ScalarMaximum(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = a.ptr[i] > val ? a.ptr[i] : val;
      }
    }

    void EwiseEq(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = (a.ptr[i] == b.ptr[i]);
      }
    }

    void ScalarEq(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = (a.ptr[i] == val);
      }
    }

    void EwiseGe(const AlignedArray &a, const AlignedArray &b, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = (a.ptr[i] >= b.ptr[i]);
      }
    }

    void ScalarGe(const AlignedArray &a, scalar_t val, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = (a.ptr[i] >= val);
      }
    }

    void EwiseLog(const AlignedArray &a, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = log(a.ptr[i]);
      }
    }

    void EwiseExp(const AlignedArray &a, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = exp(a.ptr[i]);
      }
    }

    void EwiseTanh(const AlignedArray &a, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = tanh(a.ptr[i]);
      }
    }

    void EwiseSigmoid(const AlignedArray &a, AlignedArray *out)
    {
      for (size_t i = 0; i < a.size; i++)
      {
        out->ptr[i] = 1 / (exp(-a.ptr[i]) + 1);
      }
    }
    /// END YOUR SOLUTION

    void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m, uint32_t n,
                uint32_t p)
    {
      /**
       * Multiply two (compact) matrices into an output (also comapct) matrix.  For this implementation
       * you can use the "naive" three-loop algorithm.
       *
       * Args:
       *   a: compact 2D array of size m x n
       *   b: comapct 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   m: rows of a / out
       *   n: columns of a / rows of b
       *   p: coolumns of b / out
       */

      /// BEGIN YOUR SOLUTION
      for (size_t i = 0; i < m; i++)
      {
        for (size_t j = 0; j < p; j++)
        {
          out->ptr[i * p + j] = 0;
          for (size_t k = 0; k < n; k++)
          {
            out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
          }
        }
      }
      /// END YOUR SOLUTION
    }

    inline void AlignedDot(const float *__restrict a, const float *__restrict b, float *__restrict out)
    {
      /**
       * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
       * the result to the existing out, which you should not set to zero beforehand).  We are including
       * the compiler flags here that enable the compile to properly use vector operators to implement
       * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
       * out don't have any overlapping memory (which is necessary in order for vector operations to be
       * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
       * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
       * compiler that the input array siwll be aligned to the appropriate blocks in memory, which also
       * helps the compiler vectorize the code.
       *
       * Args:
       *   a: compact 2D array of size TILE x TILE
       *   b: compact 2D array of size TILE x TILE
       *   out: compact 2D array of size TILE x TILE to write to
       */
      a = (const float *)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
      b = (const float *)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
      out = (float *)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

      /// BEGIN YOUR SOLUTION
      for (size_t i = 0; i < TILE; i++)
      {
        for (size_t j = 0; j < TILE; j++)
        {
          for (size_t k = 0; k < TILE; k++)
          {
            out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
          }
        }
      }
      // END YOUR SOLUTION
    }

    void MatmulTiled(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m,
                     uint32_t n, uint32_t p)
    {
      /**
       * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
       * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
       *   a[m/TILE][n/TILE][TILE][TILE]
       * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
       * function should call `AlignedDot()` implemented above).
       *
       * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
       * assume that this division happens without any remainder.
       *
       * Args:
       *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
       *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
       *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
       *   m: rows of a / out
       *   n: columns of a / rows of b
       *   p: coolumns of b / out
       *
       */
      /// BEGIN YOUR SOLUTION
      const size_t tile_num = TILE * TILE;
      for (size_t i = 0; i < m / TILE; i++)
      {
        for (size_t j = 0; j < p / TILE; j++)
        {
          size_t offset_out = i * TILE * p + j * tile_num;
          float temp[tile_num] = {0};
          for (size_t k = 0; k < n / TILE; k++)
          {
            size_t offset_a = i * TILE * n + k * tile_num;
            size_t offset_b = k * TILE * p + j * tile_num;
            AlignedDot(a.ptr + offset_a, b.ptr + offset_b, temp);
          }

          for (size_t idx = 0; idx < tile_num; idx++)
          {
            out->ptr[offset_out + idx] = temp[idx];
          }
        }
      }
      /// END YOUR SOLUTION
    }

    void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking maximum over `reduce_size` contiguous blocks.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */

      /// BEGIN YOUR SOLUTION
      for (size_t i = 0; i < a.size; i += reduce_size)
      {
        scalar_t max = a.ptr[i];
        for (size_t j = 0; j < reduce_size; j++)
        {
          scalar_t temp = a.ptr[i + j];
          if (temp > max)
            max = temp;
        }
        out->ptr[i / reduce_size] = max;
      }
      /// END YOUR SOLUTION
    }

    void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking sum over `reduce_size` contiguous blocks.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */

      /// BEGIN YOUR SOLUTION
      for (size_t i = 0; i < a.size; i += reduce_size)
      {
        scalar_t sum = 0;
        for (size_t j = 0; j < reduce_size; j++)
        {
          sum += a.ptr[i + j];
        }
        out->ptr[i / reduce_size] = sum;
      }
      /// END YOUR SOLUTION
    }

  } // namespace cpu
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset); });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray *out)
        { std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);
  m.def("ewise_sigmoid", EwiseSigmoid);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
