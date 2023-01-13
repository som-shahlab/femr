#include "csv.hh"

#include <fstream>

#include "absl/strings/substitute.h"
#include "zstd.h"

namespace {
const int BUFFER_SIZE = 1024 * 1024;
}  // namespace

void ZSTDCFree::operator()(ZSTD_CStream* ptr) { ZSTD_freeCStream(ptr); }

ZstdWriter::ZstdWriter(const boost::filesystem::path& filename)
    : fname(filename) {
    f.rdbuf()->pubsetbuf(nullptr, 0);
    f.open(filename.c_str());

    stream.reset(ZSTD_createCStream());
    ZSTD_initCStream(stream.get(), 1);

    in_buffer_data.resize(BUFFER_SIZE * 2);
    out_buffer_data.resize(ZSTD_compressBound(BUFFER_SIZE));

    in_buffer_pos = 0;
}

void ZstdWriter::add_data(std::string_view data) {
    if (data.size() > BUFFER_SIZE) {
        throw std::runtime_error(
            "Cannot process data greater than BUFFER_SIZE");
    }
    if (in_buffer_pos + data.size() > (BUFFER_SIZE * 2)) {
        throw std::runtime_error("Should never happen, buffsize failure");
    }
    std::memcpy(in_buffer_data.data() + in_buffer_pos, data.data(),
                data.size());
    in_buffer_pos += data.size();
    if (in_buffer_pos >= BUFFER_SIZE) {
        flush();
    }
}

void ZstdWriter::flush(bool final) {
    ZSTD_EndDirective op;
    if (final) {
        op = ZSTD_e_end;
    } else {
        op = ZSTD_e_continue;
    }

    ZSTD_inBuffer in_buffer = {in_buffer_data.data(), in_buffer_pos, 0};
    ZSTD_outBuffer out_buffer = {out_buffer_data.data(), BUFFER_SIZE, 0};

    int ret = ZSTD_compressStream2(stream.get(), &out_buffer, &in_buffer, op);

    if (ret != 0) {
        throw std::runtime_error("A single one should always be good enough");
    }

    f.write(out_buffer_data.data(), out_buffer.pos);
    std::memmove(in_buffer_data.data(), in_buffer_data.data() + in_buffer.pos,
                 in_buffer_pos - in_buffer.pos);

    in_buffer_pos -= in_buffer.pos;
}

ZstdWriter::~ZstdWriter() { flush(true); }

TextWriter::TextWriter(const boost::filesystem::path& filename)
    : fname(filename) {
    f.rdbuf()->pubsetbuf(nullptr, 0);
    f.open(filename.c_str());
}

void TextWriter::add_data(std::string_view data) {
    f.write(data.data(), data.size());
}

void ZSTDDFree::operator()(ZSTD_DStream* ptr) { ZSTD_freeDStream(ptr); }

ZstdReader::ZstdReader(const boost::filesystem::path& filename)
    : fname(filename), reader(filename) {
    stream.reset(ZSTD_createDStream());
    ZSTD_initDStream(stream.get());

    out_buffer_data.resize(BUFFER_SIZE);
    out_buffer_end = 0;

    seek(0);
}

std::string_view ZstdReader::get_data() const {
    return std::string_view(out_buffer_data.data(), out_buffer_end);
}

void ZstdReader::seek(size_t seek_amount) {
    std::memmove(out_buffer_data.data(), out_buffer_data.data() + seek_amount,
                 out_buffer_end - seek_amount);

    out_buffer_end -= seek_amount;

    if (!reader.eof()) {
        ZSTD_inBuffer in_buffer = {.src = reader.get_data().data(),
                                   .size = reader.get_data().size(),
                                   .pos = 0};
        ZSTD_outBuffer out_buffer = {.dst = out_buffer_data.data(),
                                     .size = out_buffer_data.size(),
                                     .pos = out_buffer_end};
        size_t ret =
            ZSTD_decompressStream(stream.get(), &out_buffer, &in_buffer);
        if (ZSTD_isError(ret) != 0) {
            throw std::runtime_error("Got error while decompressing? " +
                                     std::string(ZSTD_getErrorName(ret)));
        }

        reader.seek(in_buffer.pos);

        out_buffer_end = out_buffer.pos;
    }
}

bool ZstdReader::eof() const { return reader.eof() && out_buffer_end == 0; }

TextReader::TextReader(const boost::filesystem::path& filename)
    : fname(filename) {
    if (!boost::filesystem::is_regular_file(filename)) {
        throw std::runtime_error(
            absl::StrCat(filename.string(), " is not a regular file"));
    }

    f.rdbuf()->pubsetbuf(nullptr, 0);
    f.open(fname.c_str());

    buffer_data.resize(BUFFER_SIZE);
    buffer_end = 0;
    seek(0);
}

std::string_view TextReader::get_data() const {
    return std::string_view(buffer_data.data(), buffer_end);
}

void TextReader::seek(size_t seek_amount) {
    std::memmove(buffer_data.data(), buffer_data.data() + seek_amount,
                 buffer_end - seek_amount);

    buffer_end -= seek_amount;

    if (!f.eof()) {
        f.read(buffer_data.data() + buffer_end,
               buffer_data.size() - buffer_end);
        buffer_end += f.gcount();
    }
}

bool TextReader::eof() const { return buffer_end == 0 && f.eof(); }
