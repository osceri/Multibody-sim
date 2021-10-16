#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <ft2build.h>
#include <freetype/freetype.h>
#include <FreeImage.h>

#include <stdio.h>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <stack>



union vec4 {
	float v[4];
	__m128 m;
};

union mat4 {
	float v[16];
	vec4 c[4];
	__m128 m[4];
};

void print_vec4(vec4 v) {
	printf("[%.2f, %.2f, %.2f, %.2f]\n", v.v[0], v.v[1], v.v[2], v.v[3]);
}

void print_mat4(mat4 v) {
	printf("/%.2f, %.2f, %.2f, %.2f\\\n", v.v[0], v.v[4], v.v[8], v.v[12]);
	printf("|%.2f, %.2f, %.2f, %.2f|\n", v.v[1], v.v[5], v.v[9], v.v[13]);
	printf("|%.2f, %.2f, %.2f, %.2f|\n", v.v[2], v.v[6], v.v[10], v.v[14]);
	printf("\\%.2f, %.2f, %.2f, %.2f/\n", v.v[3], v.v[7], v.v[11], v.v[15]);
}

vec4 cross(vec4 A, vec4 B) {
	vec4 H;
	H.m = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(A.m, B.m, _MM_SHUFFLE(0, 1, 2, 1)), _mm_shuffle_ps(B.m, A.m, _MM_SHUFFLE(0, 0, 0, 2))), _mm_mul_ps(_mm_shuffle_ps(A.m, B.m, _MM_SHUFFLE(0, 0, 0, 2)), _mm_shuffle_ps(B.m, A.m, _MM_SHUFFLE(0, 1, 2, 1))));
	return H;
}

vec4 dot(vec4 A, vec4 B) {
	vec4 H;
	A.m = _mm_mul_ps(A.m, B.m);
	H.m = _mm_add_ps(A.m, _mm_add_ps(_mm_shuffle_ps(A.m, A.m, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(A.m, A.m, _MM_SHUFFLE(3, 1, 0, 2))));
	return H;
}

vec4 dot_4(vec4 A, vec4 B) {
	vec4 H;
	A.m = _mm_mul_ps(A.m, B.m);
	H.m = _mm_add_ps(_mm_add_ps(A.m, _mm_shuffle_ps(A.m, A.m, _MM_SHUFFLE(0, 3, 2, 1))), _mm_add_ps(_mm_shuffle_ps(A.m, A.m, _MM_SHUFFLE(1, 0, 3, 2)), _mm_shuffle_ps(A.m, A.m, _MM_SHUFFLE(2, 1, 0, 3))));
	return H;
}

mat4 transpose(mat4 A) {
	mat4 B;
	B.m[0] = _mm_shuffle_ps(A.m[0], A.m[1], _MM_SHUFFLE(1, 0, 1, 0));
	B.m[1] = _mm_shuffle_ps(A.m[0], A.m[1], _MM_SHUFFLE(3, 2, 3, 2));
	B.m[2] = _mm_shuffle_ps(A.m[2], A.m[3], _MM_SHUFFLE(1, 0, 1, 0));
	B.m[3] = _mm_shuffle_ps(A.m[2], A.m[3], _MM_SHUFFLE(3, 2, 3, 2));
	A.m[0] = _mm_shuffle_ps(B.m[0], B.m[2], _MM_SHUFFLE(2, 0, 2, 0));
	A.m[1] = _mm_shuffle_ps(B.m[0], B.m[2], _MM_SHUFFLE(3, 1, 3, 1));
	A.m[2] = _mm_shuffle_ps(B.m[1], B.m[3], _MM_SHUFFLE(2, 0, 2, 0));
	A.m[3] = _mm_shuffle_ps(B.m[1], B.m[3], _MM_SHUFFLE(3, 1, 3, 1));
	return A;
}

vec4 matvec_mul(mat4 A, vec4 B) {
	mat4 H;
	H.m[0] = _mm_shuffle_ps(B.m, B.m, _MM_SHUFFLE(0, 0, 0, 0));
	H.m[1] = _mm_shuffle_ps(B.m, B.m, _MM_SHUFFLE(1, 1, 1, 1));
	H.m[2] = _mm_shuffle_ps(B.m, B.m, _MM_SHUFFLE(2, 2, 2, 2));
	H.m[3] = _mm_shuffle_ps(B.m, B.m, _MM_SHUFFLE(3, 3, 3, 3));
	B.m = _mm_add_ps(_mm_add_ps(_mm_mul_ps(A.m[0], H.m[0]), _mm_mul_ps(A.m[1], H.m[1])), _mm_add_ps(_mm_mul_ps(A.m[2], H.m[2]), _mm_mul_ps(A.m[3], H.m[3])));
	return B;
}

mat4 matmat_mul(mat4 A, mat4 B) {
	mat4 H;
	H.c[0] = matvec_mul(A, B.c[0]);
	H.c[1] = matvec_mul(A, B.c[1]);
	H.c[2] = matvec_mul(A, B.c[2]);
	H.c[3] = matvec_mul(A, B.c[3]);
	return H;
}

mat4 projection_matrix(float w, float h, float f, float n) {
	mat4 P = {
		n / w, 0., 0., 0.,
		0., n / h, 0., 0.,
		0., 0., (f + n) / (n - f), -1.,
		0., 0., 2.*f*n / (n - f), 0.
	};
	return P;
}

mat4 orthographic_matrix(float w, float h, float f, float n) {
	mat4 P = {
		2. / w, 0., 0., 0.,
		0., 2. / h, 0., 0.,
		0., 0., 0., 0.,
		-1., -1., 0., 1.
	};
	return P;
}

mat4 yaw_matrix(float yaw) {
	mat4 P = {
		cosf(yaw), 0., sinf(yaw), 0.,
		0., 1., 0., 0.,
		-sinf(yaw), 0., cosf(yaw), 0.,
		0., 0., 0., 1.,
	};
	return P;
}

mat4 roll_matrix(float roll) {
	mat4 P = {
		cosf(roll), sinf(roll), 0., 0.,
		-sinf(roll), cosf(roll), 0., 0.,
		0., 0., 1., 0.,
		0., 0., 0., 1.,
	};
	return P;
}

mat4 pitch_matrix(float pitch) {
	mat4 P = {
		1., 0., 0., 0.,
		0., cosf(pitch), sinf(pitch), 0.,
		0., -sinf(pitch), cosf(pitch), 0.,
		0., 0., 0., 1.,
	};
	return P;
}

mat4 translation_matrix(float x, float y, float z) {
	mat4 P = {
		1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.,
		x, y, z, 1.,
	};
	return P;
}

mat4 scale_matrix(float x, float y, float z) {
	mat4 P = {
		x, 0., 0., 0.,
		0., y, 0., 0.,
		0., 0., z, 0.,
		0., 0., 0., 1.,
	};
	return P;
}

mat4 identity_matrix() {
	mat4 P = {
		1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.,
		0., 0., 0., 1.,
	};
	return P;
}


union pixel {
	int p;
	char agbr[4];
	float f;
};

#define WRAP_EDGE				0
#define WRAP_BORDER				1
#define WRAP_REPEAT				2
#define WRAP_MIRRORED_REPEAT	3


class shader {
	uint32_t id;

public:
	shader(uint32_t type, std::string source) : id(-1) {
		make_shader(type, source);
	}

	shader() : id(-1) {

	}

	~shader() {
		if (id != -1) glDeleteShader(id);
		id = -1;
	}

	void make_shader(uint32_t type, std::string source) {
		if (id != -1) glDeleteShader(id);
		id = -1;

		id = glCreateShader(type);
		const char* c_source = source.c_str();
		int source_length = source.size();
		glShaderSource(id, 1, &c_source, &source_length);
		glCompileShader(id);

		GLint shader_compiled;
		glGetShaderiv(id, GL_COMPILE_STATUS, &shader_compiled);
		if (shader_compiled != GL_TRUE)
		{
			GLsizei log_length = 0;
			GLchar message[1024];
			glGetShaderInfoLog(id, 1024, &log_length, message);
			printf(message);
		}
	}

	uint32_t get_id() {
		return id;
	}

};

std::unordered_map<std::string, uint32_t> programs;
class program {
	uint32_t id;

public:
	program() : id(-1) {

	}

	program(const char* _name) : id(-1) {
		make_program(_name);

	}

	~program() {
		if (id != -1) {
			glDeleteProgram(id);
			id = -1;
		}
	}

	void make_program(const char* _name) {
		int map_id = 0;

		if (id != -1) {
			glDeleteProgram(id);
			id = -1;
		}

		auto find = programs.find(_name);
		if (find != programs.end()) {
			if (glIsProgram(find->second)) {
				id = find->second;
				return; // Exit if the shader already exists
			}
			else {
				programs.erase(find);
				map_id = 1;
			}
		}
		else {
			map_id = 1;
		}

		std::unordered_map<int, shader> shaders;

		std::filebuf fb;
		char file_path[256];
		sprintf_s(file_path, "Programs/%s.shader", _name);
		fb.open(file_path, std::ios::in);
		std::istream is(&fb);

		std::string line;
		std::stringstream ss;
		std::string source;
		uint32_t type = 0;

		struct { const char str[30]; int type; } f_list[] = {
			{"//??GL_VERTEX_SHADER", GL_VERTEX_SHADER},
			{"//??GL_FRAGMENT_SHADER", GL_FRAGMENT_SHADER},
			{"//??GL_GEOMETRY_SHADER", GL_GEOMETRY_SHADER},
			{"//??GL_TESS_EVALUATION_SHADER", GL_TESS_EVALUATION_SHADER},
			{"//??GL_TESS_CONTROL_SHADER", GL_TESS_CONTROL_SHADER},
			{"//??GL_COMPUTE_SHADER", GL_COMPUTE_SHADER},
		};

		int any = 0;
		while (std::getline(is, line)) {
			for (auto i : f_list) {
				if (line == i.str) {
					any = 1;
					if (type != 0) {
						source = ss.str();
						shaders[type].make_shader(type, source);
						ss.str(std::string());
					}
					type = i.type;
					any = 1;
					break;
				}
			}
			if (any) {
				any = 0;
			}
			else {
				ss << line << '\n';
			}
		}

		if (type != 0) {
			source = ss.str();
			shaders[type].make_shader(type, source);
		}

		fb.close();

		id = glCreateProgram();

		for (auto shader : shaders) {
			glAttachShader(id, shader.second.get_id());
		}

		glLinkProgram(id);

		GLint program_linked;
		glGetProgramiv(id, GL_LINK_STATUS, &program_linked);
		if (program_linked != GL_TRUE)
		{
			GLsizei log_length = 0;
			GLchar message[1024];
			glGetProgramInfoLog(id, 1024, &log_length, message);
			printf(message);
		}

		if (map_id) {
			programs[_name] = id;
		}

	}

	uint32_t get_id() {
		return id;
	}

	uint32_t get_uniform_location(const char* uniform) {
		return glGetUniformLocation(id, uniform);
	}

};

std::unordered_map<std::string, uint32_t> textures;
class texture {
	uint32_t id;

public:

	texture(const char* _name) : id(-1) {
		make_texture(_name);
	}

	~texture() {
		if (id != -1) {
			glDeleteTextures(1, &id);
			id = -1;
		}
	}

	void make_texture(const char* _name) {
		int map_id = 0;

		if (id != -1) {
			glDeleteTextures(1, &id);
			id = -1;
		}


		auto find = textures.find(_name);
		if (find != textures.end()) {
			if (glIsTexture(find->second)) {
				id = find->second;
			}
			else {
				textures.erase(find);
				map_id = 1;
			}
		}
		else {
			map_id = 1;
		}

		char path[256];
		sprintf(path, "textures/%s", _name);
		FIBITMAP* frdata;
		frdata = FreeImage_Load(FIF_UNKNOWN, path, 0);

		glGenTextures(1, &id);
		glBindTexture(GL_TEXTURE_2D, id);
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGBA,
			FreeImage_GetWidth(frdata),
			FreeImage_GetHeight(frdata),
			0,
			GL_RGBA,
			GL_UNSIGNED_INT,
			FreeImage_GetBits(frdata)
		);
		// set texture options
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);

		if (map_id) {
			textures[_name] = id;
		}
	}

	uint32_t get_id() {
		return id;
	}

};


struct _planet_vertex {
	float nx, ny, nz;
	float u, v;
};

_planet_vertex interpolate_planet_vertex(_planet_vertex a, _planet_vertex b) {
	float x = a.nx + b.nx;
	float y = a.ny + b.ny;
	float z = a.nz + b.nz;
	float N = 1. / sqrtf(x*x + y * y + z * z);
	float u = 0.5*(a.u + b.u);
	float v = 0.5*(a.v + b.v);
	return { N*x, N*y, N*z, u, v };
}

struct _triangle {
	int i, j, k;
public:
	_triangle() : i(0), j(0), k(0) {

	}
	_triangle(int _i, int _j, int _k) : i(_i), j(_j), k(_k) {

	}
};

struct _edge {
	int i, j;
public:
	_edge() : i(0), j(0) {
	}
	_edge(int _i, int _j) {
		if (_j < _i) {
			i = _i;
			j = _j;
		}
		else {
			i = _j;
			j = _i;
		}
	}

	bool operator()(const _edge& l, const _edge& r) const {
		bool ret = 0;
		if (l.i < r.i)
			return true;
		if (l.i > r.i)
			return false;
		if (l.j < r.j)
			return true;

		return false;
	}
};



class planet {
	uint32_t partitions;
	std::vector<_planet_vertex> data;
	std::vector<_triangle> triangles;
	uint32_t data_id;
	uint32_t indice_id;
	uint32_t vao;

	program planet_shader;
	program sun_shader;

	float x, y, z, r;

	float radius;

public:
	planet() : partitions(3), radius(10), data_id(-1) {
		x = 0.;
		y = 0.;
		z = 1.;
		r = 1.;

			std::map<_edge, int, _edge> unique_vertices;
			std::vector<_triangle> _triangles[2];
			
			_planet_vertex init_data[] = {
				{ 0., -1.,  0.,		0.0, 0.0 },
				{ 0.,  0., -1.,		0.5, 0.0 },
				{ 0., -1.,  0.,		1.0, 0.0 },
				{-1.,  0.,  0.,		0.0, 0.5 },
				{ 0.,  1.,  0.,		0.5, 0.5 },
				{ 1.,  0.,  0.,		1.0, 0.5 },
				{ 0., -1.,  0.,		0.0, 1.0 },
				{ 0.,  0.,  1.,		0.5, 1.0 },
				{ 0., -1.,  0.,		1.0, 1.0 },
			};

			_triangle init_triangles[] = {
				{0, 3, 1,},
				{3, 4, 1,},
				{1, 5, 2,},
				{1, 4, 5,},
				{3, 7, 4,},
				{3, 6, 7,},
				{4, 7, 5,},
				{7, 8, 5,},
			};

			for (auto g : init_data) data.push_back(g);

			int p = 0;
			for (auto g : init_triangles) _triangles[p % 2].push_back(g);

			for (; p < partitions; p++) {
				for (auto t : _triangles[p % 2]) {
					int a = unique_vertices[_edge{ t.i, t.j }];
					if (a == 0) {
						a = data.size();
						unique_vertices[_edge{ t.i, t.j }] = a;
						data.push_back(interpolate_planet_vertex(data[t.i], data[t.j]));
					}
					int b = unique_vertices[_edge{ t.j, t.k }];
					if (b == 0) {
						b = data.size();
						unique_vertices[_edge{ t.j, t.k }] = b;
						data.push_back(interpolate_planet_vertex(data[t.j], data[t.k]));
					}
					int c = unique_vertices[_edge{ t.k, t.i }];
					if (c == 0) {
						c = data.size();
						unique_vertices[_edge{ t.k, t.i }] = c;
						data.push_back(interpolate_planet_vertex(data[t.k], data[t.i]));
					}

					_triangles[(p + 1) % 2].push_back(_triangle{ t.i, a, c });
					_triangles[(p + 1) % 2].push_back(_triangle{ c, a, b });
					_triangles[(p + 1) % 2].push_back(_triangle{ b, a, t.j });
					_triangles[(p + 1) % 2].push_back(_triangle{ c, b, t.k });
				}

				_triangles[p % 2].clear();
			}

			for (auto t : _triangles[p % 2]) triangles.push_back(t);


		planet_shader.make_program("planet");
		sun_shader.make_program("sun");

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &data_id);
		glBindBuffer(GL_ARRAY_BUFFER, data_id);
		glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(_planet_vertex), data.data(), GL_STATIC_DRAW);

		glGenBuffers(1, &indice_id);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indice_id);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.size() * sizeof(_triangle), triangles.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(_planet_vertex), (const void*)0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(_planet_vertex), (const void*)12);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}

	~planet() {

	}

	void draw_planet(vec4 light, float time, float pitch, float yaw, float x, float y, float z, float radius) {
		glUseProgram(planet_shader.get_id());

		mat4 m = scale_matrix(radius, radius, radius);
		mat4 R = matmat_mul(pitch_matrix(pitch), roll_matrix(yaw));
		mat4 v = matmat_mul(R, translation_matrix(x, y, z));
		mat4 mv = matmat_mul(v, m);
		mat4 p = projection_matrix(.2, .2*1080. / 1920., 1000, 0.1);

		glUniformMatrix4fv(planet_shader.get_uniform_location("mv"), 1, 0, mv.v);
		glUniformMatrix4fv(planet_shader.get_uniform_location("p"), 1, 0, p.v);
		glUniform4fv(planet_shader.get_uniform_location("light"), 1, light.v);
		glUniform1f(planet_shader.get_uniform_location("time"), time);
		glUniform3f(planet_shader.get_uniform_location("colour"), 0.3*(sinf(radius)+1), 0.3*(sinf(radius/2)+1), 0.5*(tanf(radius)+1));

		glBindVertexArray(vao);

		glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, NULL);

		glBindVertexArray(0);

		glUseProgram(0);

	}

	void draw_sun(float time, float pitch, float yaw, float x, float y, float z, float radius) {
		glUseProgram(sun_shader.get_id());

		mat4 m = scale_matrix(radius, radius, radius);
		mat4 R = matmat_mul(pitch_matrix(pitch), roll_matrix(yaw));
		mat4 v = matmat_mul(R, translation_matrix(x, y, z));
		mat4 mv = matmat_mul(v, m);
		mat4 p = projection_matrix(.2, .2*1080. / 1920., 1000, 0.1);

		glUniformMatrix4fv(sun_shader.get_uniform_location("mv"), 1, 0, mv.v);
		glUniformMatrix4fv(sun_shader.get_uniform_location("p"), 1, 0, p.v);
		glUniform1f(sun_shader.get_uniform_location("time"), time);
		
		glBindVertexArray(vao);

		glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, NULL);

		glBindVertexArray(0);

		glUseProgram(0);

	}

};


struct Character {
	unsigned int TextureID;               // ID handle of the glyph texture
	struct { int32_t x, y; } size, bearing, advance; // Size of glyph, offset from baseline to left/top of glyph, offset to advance to next glyph
};


class UI {
	float height;
	float width;
	float* height_rel;
	float* width_rel;

	uint32_t data_id;
	uint32_t vao;

	FT_Library ft;
	FT_Face face;
	program font_shader;
	program image_shader;
	program clr_shader;

	std::map<char, Character> Characters;

public:
	UI(float* _width, float* _height) : width(*_width), height(*_height), width_rel(_width), height_rel(_height), data_id(-1), vao(-1) {
	}

	void initialize() {
		font_shader.make_program("font");
		image_shader.make_program("image");
		clr_shader.make_program("clr");

		init_characters();
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &data_id);
		glBindBuffer(GL_ARRAY_BUFFER, data_id);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glUseProgram(0);
	}

	~UI() {
		if (data_id != -1) {
			glDeleteBuffers(1, &data_id);
			data_id = -1;
		}
		if (vao != -1) {
			glDeleteVertexArrays(1, &vao);
			vao = -1;
		}
	}

	float get_width() {
		return width;
	}

	float get_width_rel() {
		return *width_rel;
	}

	float get_height() {
		return height;
	}

	float get_height_rel() {
		return *height_rel;
	}

	void init_characters() {
		glUseProgram(font_shader.get_id());

		if (FT_Init_FreeType(&ft))
		{
			printf("ERROR::FREETYPE: Could not init FreeType Library");
		}

		if (FT_New_Face(ft, "fonts/tahoma.ttf", 0, &face))
		{
			printf("ERROR::FREETYPE: Failed to load font");
		}

		FT_Set_Pixel_Sizes(face, 0, 12);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // disable byte-alignment restriction

		for (unsigned char c = 0; c < 200; c++)
		{
			// load character glyph 
			if (FT_Load_Char(face, c, FT_LOAD_RENDER))
			{
				printf("ERROR::FREETYTPE: Failed to load Glyph");
				continue;
			}

			// generate texture
			unsigned int texture;
			glGenTextures(1, &texture);
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexImage2D(
				GL_TEXTURE_2D,
				0,
				GL_RED,
				face->glyph->bitmap.width,
				face->glyph->bitmap.rows,
				0,
				GL_RED,
				GL_UNSIGNED_BYTE,
				face->glyph->bitmap.buffer
			);
			// set texture options
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			// now store character for later use
			Character character;

			character.TextureID = texture;
			character.size.x = face->glyph->bitmap.width;
			character.size.y = face->glyph->bitmap.rows;
			character.bearing.x = face->glyph->bitmap_left;
			character.bearing.y = face->glyph->bitmap_top;
			character.advance.x = face->glyph->advance.x;
			character.advance.y = face->glyph->advance.y;

			Characters.insert(std::pair<char, Character>(c, character));
		}
		glBindTexture(GL_TEXTURE_2D, 0);

		FT_Done_Face(face);
		FT_Done_FreeType(ft);
		
	}

	// returns envelope
	float envelope_line(std::string text, float scale)
	{
		float dx = 0.;

		for (char c : text) {
			Character ch = Characters[c];
			if (c == '\n') {
				continue;
			}

			dx += (ch.advance.x >> 6) * scale;
		}
		
		return dx;
	}

	vec4 envelope_text(std::string text, float scale) {
		float dy = 0;
		float dx = 0;
		vec4 envelope = vec4{ 0, 0, 0, 0 };
		std::vector<std::string> lines;
		std::string line;
		for (char c : text) {
			if (c == '\n') {
				float nx = envelope_line(line, scale);
				if (dx < nx) dx = nx;
				dy -= 14 * scale;
				line.clear();
				continue;
			}
			line.push_back(c);
		}
		float nx = envelope_line(line, scale);
		if (dx < nx) dx = nx;
		return vec4{0, dy, dx, 14 * scale };
	}

	void render_text(std::string text, float x, float y, float scale, vec4 rgba) {
		std::vector<std::string> lines;
		std::string line;
		for (char c : text) {
			if (c == '\n') {
				render_line(line, x, y, scale, rgba);
				y -= 14 * scale;
				line.clear();
				continue;
			}
			line.push_back(c);
		}
		render_line(line, x, y, scale, rgba);
	}
	
	void render_line(std::string text, float x, float y, float scale, vec4 rgba)
	{
		glEnable(GL_CULL_FACE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_DEPTH_TEST);

		mat4 p = orthographic_matrix(get_width_rel(), get_height_rel() , 1000, 0.1);
		glUseProgram(font_shader.get_id());
		// activate corresponding render state
		glUniform4fv(font_shader.get_uniform_location("textColor"), 1, rgba.v);
		glUniformMatrix4fv(font_shader.get_uniform_location("projection"), 1, 0, p.v);
		glUniform1i(font_shader.get_uniform_location("text"), 0);
		glActiveTexture(GL_TEXTURE0);
		glBindVertexArray(vao);

		float dx = 0.;
		float dy = 0.;

		for (char c : text) {
			Character ch = Characters[c];
			if (c == '\n') {
				continue;
			}

			float xpos = x + dx + ch.bearing.x * scale;
			float ypos = y - dy - (ch.size.y - ch.bearing.y) * scale;

			float w = ch.size.x * scale;
			float h = ch.size.y * scale;

			float vertices[6][4] = {
				{ xpos,     ypos + h,   0.0f, 0.0f },
				{ xpos,     ypos,       0.0f, 1.0f },
				{ xpos + w, ypos,       1.0f, 1.0f },
				{ xpos,     ypos + h,   0.0f, 0.0f },
				{ xpos + w, ypos,       1.0f, 1.0f },
				{ xpos + w, ypos + h,   1.0f, 0.0f }
			};

			glBindTexture(GL_TEXTURE_2D, ch.TextureID);

			glBindBuffer(GL_ARRAY_BUFFER, data_id);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glDrawArrays(GL_TRIANGLES, 0, 6);

			dx += (ch.advance.x >> 6) * scale;
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindVertexArray(0);
		glUseProgram(0);


		glEnable(GL_DEPTH_TEST);

	}

	void render_box(float x_a, float y_a, float x_b, float y_b, float x_c, float y_c, float x_d, float y_d, vec4 rgba)
	{
		glEnable(GL_CULL_FACE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDisable(GL_DEPTH_TEST);

		mat4 p = orthographic_matrix(*width_rel, *height_rel, 1000, 0.1);
		glUseProgram(clr_shader.get_id());
		// activate corresponding render state
		glUniform4fv(clr_shader.get_uniform_location("Color"), 1, rgba.v);
		glUniformMatrix4fv(clr_shader.get_uniform_location("projection"), 1, 0, p.v);
		glBindVertexArray(vao);


		float vertices[6][4] = {
			{ x_a, y_a,   0.0f, 0.0f },
			{ x_b, y_b,   0.0f, 1.0f },
			{ x_c, y_c,   1.0f, 1.0f },
			{ x_a, y_a,   0.0f, 0.0f },
			{ x_c, y_c,   1.0f, 1.0f },
			{ x_d, y_d,   1.0f, 0.0f }
		};

		glBindBuffer(GL_ARRAY_BUFFER, data_id);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindVertexArray(0);
		glUseProgram(0);


		glEnable(GL_DEPTH_TEST);
	}

	uint32_t UI_func() {

	}
};


enum WIDGET_ALIGNMENT_VERTICAL {
	WIDGET_ALIGNMENT_BOTTOM_V = 0,
	WIDGET_ALIGNMENT_TOP_V = 1,
	WIDGET_ALIGNMENT_CENTER_V = 2,
};

enum WIDGET_ALIGNMENT_HORIZONTAL {
	WIDGET_ALIGNMENT_LEFT_H = 0,
	WIDGET_ALIGNMENT_RIGHT_H = 1,
	WIDGET_ALIGNMENT_CENTER_H = 2,
};

enum WIDGET_BORDER_ALIGNMENT {
	WIDGET_BORDER_OUTSIDE = 0,
	WIDGET_BORDER_CENTER = 1,
	WIDGET_BORDER_INSIDE = 2,
};

enum WIDGET_TEXT_ALIGNMENT_HORIZONTAL {
	WIDGET_TEXT_ALIGNMENT_LEFT_H = 0,
	WIDGET_TEXT_ALIGNMENT_RIGHT_H = 1,
	WIDGET_TEXT_ALIGNMENT_CENTER_H = 2,
};

enum WIDGET_TEXT_ALIGNMENT_VERTICAL {
	WIDGET_TEXT_ALIGNMENT_TOP_V = 0,
	WIDGET_TEXT_ALIGNMENT_BOTTOM_V = 1,
	WIDGET_TEXT_ALIGNMENT_CENTER_V = 2,
};

enum WIDGET_FLAGS {
	WIDGET_FLAG_NORMAL = 0,
	WIDGET_FLAG_DROPDOWN_DELTA = 1,
};

enum WIDGET_RETURN_STATE {
	WIDGET_RETURN_STATE_PASSIVE = 0,
	WIDGET_RETURN_STATE_HOVER = 1,
	WIDGET_RETURN_STATE_ACTIVE = 2,
	WIDGET_RETURN_STATE_PRESSED = 3,
	WIDGET_RETURN_STATE_RELEASED = 4,
	WIDGET_RETURN_STATE_LOCKED = 5,
};

enum WIDGET_STATE {
	WIDGET_STATE_UNLOCKED = 0,
	WIDGET_STATE_LOCKED = 1,
	WIDGET_STATE_FLIPPABLE = 2,
	WIDGET_STATE_DISAPPEARABLE = 3,
};


#define COLOR_EXTRA_BLACK	vec4{0.0, 0.0, 0.0, 1.0}
#define COLOR_BLACK			vec4{0.1, 0.1, 0.1, 1.0}
#define COLOR_DARK_GRAY		vec4{0.3, 0.3, 0.3, 1.0}
#define COLOR_GRAY			vec4{0.5, 0.5, 0.5, 1.0}
#define COLOR_LIGHT_GRAY	vec4{0.7, 0.7, 0.7, 1.0}
#define COLOR_WHITE			vec4{0.9, 0.9, 0.9, 1.0}

#define COLOR_THEME			vec4{0.35, 0.40, 0.88, 1.0}
#define COLOR_GRAY_0		vec4{0.0, 0.0, 0.0, 1.0}
#define COLOR_GRAY_1		vec4{0.1, 0.1, 0.1, 1.0}
#define COLOR_GRAY_2		vec4{0.2, 0.2, 0.2, 1.0}
#define COLOR_GRAY_3		vec4{0.3, 0.3, 0.3, 1.0}
#define COLOR_GRAY_4		vec4{0.4, 0.4, 0.4, 1.0}
#define COLOR_GRAY_5		vec4{0.5, 0.5, 0.5, 1.0}
#define COLOR_GRAY_6		vec4{0.6, 0.6, 0.6, 1.0}
#define COLOR_GRAY_7		vec4{0.7, 0.7, 0.7, 1.0}
#define COLOR_GRAY_8		vec4{0.8, 0.8, 0.8, 1.0}
#define COLOR_GRAY_9		vec4{0.9, 0.9, 0.9, 1.0}
#define COLOR_BLACK			vec4{1.0, 1.0, 1.0, 1.0}


class Widget {
public:
	struct {
		float left, right, up, down;
	} pos; // Pixel position of edges of widget

	struct {
		float left, right, up, down;
	} border; // Border widths

	std::string text;
	struct {
		float left, right, up, down;
	} text_alignment; // text alignment to widget

	struct {
		vec4 passive, active, hover, locked;
	} background_color, border_color, text_color;

	WIDGET_TEXT_ALIGNMENT_HORIZONTAL text_alignment_horizontal_type;
	WIDGET_TEXT_ALIGNMENT_VERTICAL text_alignment_vertical_type;
	WIDGET_BORDER_ALIGNMENT border_alignment_type;
	WIDGET_ALIGNMENT_HORIZONTAL alignment_horizontal_type;
	WIDGET_ALIGNMENT_VERTICAL alignment_vertical_type;
	WIDGET_FLAGS flag_type;
	WIDGET_STATE state_type;


	Widget() :
		pos{ 0., 0., 0., 0. }, border{ 0., 0., 0., 0. }, border_alignment_type(WIDGET_BORDER_OUTSIDE), flag_type(WIDGET_FLAG_NORMAL), state_type(WIDGET_STATE_UNLOCKED),
		alignment_horizontal_type(WIDGET_ALIGNMENT_LEFT_H), alignment_vertical_type(WIDGET_ALIGNMENT_CENTER_V),
		text(""), text_alignment{ 0., 0., 0., 0. }, text_alignment_horizontal_type(WIDGET_TEXT_ALIGNMENT_CENTER_H), text_alignment_vertical_type(WIDGET_TEXT_ALIGNMENT_CENTER_V),
		background_color{ COLOR_GRAY_2, COLOR_THEME, COLOR_GRAY_3, COLOR_GRAY_2 }, border_color{ COLOR_GRAY_6, COLOR_GRAY_4, COLOR_GRAY_8, COLOR_GRAY_6 }, text_color{ COLOR_GRAY_8, COLOR_GRAY_7, COLOR_GRAY_9, COLOR_GRAY_6 }
	{}

	~Widget() {}

	void set_pos(float left, float right, float down, float up) {
		pos.left = left;
		pos.right = right;
		pos.down = down;
		pos.up = up;
	}

	void set_border(float left, float right, float down, float up) {
		border.left = left;
		border.right = right;
		border.down = down;
		border.up = up;
	}

	void set_border_p(float left, float right, float down, float up) {
		border.left = left / (pos.right - pos.left);
		border.right = right / (pos.right - pos.left);
		border.down = down / (pos.up - pos.down);
		border.up = up / (pos.up - pos.down);
	}

	void set_alignment(float left, float right, float down, float up) {
		text_alignment.left = left;
		text_alignment.right = right;
		text_alignment.down = down;
		text_alignment.up = up;
	}

	void set_alignment_p(float left, float right, float down, float up) {
		text_alignment.left = left / (pos.right - pos.left);
		text_alignment.right = right / (pos.right - pos.left);
		text_alignment.down = down / (pos.up - pos.down);
		text_alignment.up = up / (pos.up - pos.down);
	}

	void set_below_widget(Widget* widget) {
		set_pos(0, pos.right - pos.left, 0, pos.up - pos.down);
		set_pos(widget->pos.left, widget->pos.left + pos.right, widget->pos.up, widget->pos.up + pos.up);

	}

	void set_above_widget(Widget* widget) {
		set_pos(0, pos.right - pos.left, 0, pos.up - pos.down);
		set_pos(widget->pos.left, widget->pos.left + pos.right, widget->pos.down - pos.up, widget->pos.down);
	}

	void set_beside_widget(Widget* widget) {
		set_pos(0, pos.right - pos.left, 0, pos.up - pos.down);
		set_pos(widget->pos.right, widget->pos.right + pos.right, widget->pos.down, widget->pos.down + pos.up);
	}

	void set_opposite_widget(Widget* widget) {
		set_pos(0, pos.right - pos.left, 0, pos.up - pos.down);
		set_pos(widget->pos.left - pos.right, widget->pos.left, widget->pos.down, widget->pos.down + pos.up);
	}

	int widget_p = 0;
	int widget_c = 0;
	WIDGET_RETURN_STATE draw_widget(UI* ui, uint32_t lclick, uint32_t rclick, float x, float y) {
		WIDGET_RETURN_STATE return_state = WIDGET_RETURN_STATE_PASSIVE;
		float width = ui->get_width_rel();
		float height = ui->get_height_rel();
		float left = pos.left, right = pos.right, up = pos.up, down = pos.down;
		if (alignment_horizontal_type == WIDGET_ALIGNMENT_RIGHT_H) left = width - pos.right;
		if (alignment_horizontal_type == WIDGET_ALIGNMENT_CENTER_H) left = pos.left + 0.5*(width - pos.right);
		if (alignment_horizontal_type == WIDGET_ALIGNMENT_RIGHT_H) right = width - pos.left;
		if (alignment_horizontal_type == WIDGET_ALIGNMENT_CENTER_H) right = pos.right + 0.5*(width - pos.right);
		if (alignment_vertical_type == WIDGET_ALIGNMENT_TOP_V) down = height - pos.up;
		if (alignment_vertical_type == WIDGET_ALIGNMENT_CENTER_V) down = pos.down + 0.5*(height - pos.down);
		if (alignment_vertical_type == WIDGET_ALIGNMENT_TOP_V) up = height - pos.down;
		if (alignment_vertical_type == WIDGET_ALIGNMENT_CENTER_V) up = pos.up + 0.5*(height - pos.down);

		if ((left < 0) | (down < 0) | (up > height) | (right > width)) {
			if (state_type == WIDGET_STATE_DISAPPEARABLE) {
				return WIDGET_RETURN_STATE_LOCKED;
			}
			if (state_type == WIDGET_STATE_FLIPPABLE) {
				if (right > width) {
					left = left + pos.left - pos.right;
					right = right + pos.left - pos.right;
				}
				if (down < 0) {
					up = up + pos.up - pos.down;
					down = down + pos.up - pos.down;
				}
			}
		}

		float bo = 0, bi = 0;
		if (border_alignment_type == WIDGET_BORDER_INSIDE) {
			bo = 0.0;
			bi = 1.0;
		}
		if (border_alignment_type == WIDGET_BORDER_CENTER) {
			bo = 0.5;
			bi = 0.5;
		}
		if (border_alignment_type == WIDGET_BORDER_OUTSIDE) {
			bo = 1.0;
			bi = 0.0;
		}

		float bo_left = left - bo * border.left;
		float bo_right = right + bo * border.right;
		float bo_down = down - bo * border.down;
		float bo_up = up + bo * border.up;

		float bi_left = left + bi * border.left;
		float bi_right = right - bi * border.right;
		float bi_down = down + bi * border.down;
		float bi_up = up - bi * border.up;

		float b_left = left;
		float b_right = right;
		float b_down = down;
		float b_up = up;

		vec4 env = ui->envelope_text(text, 1.0);
		float t_left = left;
		float t_right = right;
		float t_down = down;
		float t_up = up;

		vec4 bck_color = vec4{ 0., 0., 0., 0. };
		vec4 txt_color = vec4{ 0., 0., 0., 0. };
		vec4 brd_color = vec4{ 0., 0., 0., 0. };
		if (state_type != WIDGET_STATE_LOCKED) {
			if (left < x & x < right & down < (height - y) & (height - y) < up) {
				widget_p = widget_c;
				if (lclick == GLFW_PRESS) {
					widget_c = 1;
					return_state = WIDGET_RETURN_STATE_ACTIVE;
					bck_color = background_color.active;
					txt_color = text_color.active;
					brd_color = border_color.active;
				}
				else {
					widget_c = 0;
					return_state = WIDGET_RETURN_STATE_HOVER;
					bck_color = background_color.hover;
					txt_color = text_color.hover;
					brd_color = border_color.hover;
				}
				if (lclick == GLFW_RELEASE & widget_p != widget_c) {
					return_state = WIDGET_RETURN_STATE_RELEASED;
				}
				if (lclick == GLFW_PRESS & widget_p != widget_c) {
					return_state = WIDGET_RETURN_STATE_PRESSED;
				}

			}
			else {
				return_state = WIDGET_RETURN_STATE_PASSIVE;
				bck_color = background_color.passive;
				txt_color = text_color.passive;
				brd_color = border_color.passive;
			}
		}
		else {
			return_state = WIDGET_RETURN_STATE_LOCKED;
			bck_color = background_color.locked;
			txt_color = text_color.locked;
			brd_color = border_color.locked;

		}

		ui->render_box(b_left, b_up, b_left, b_down, b_right, b_down, b_right, b_up, bck_color);
		ui->render_box(bo_left, bo_up, bo_left, bo_down, bi_left, bi_down, bi_left, bi_up, brd_color);
		ui->render_box(bo_left, bo_up, bi_left, bi_up, bi_right, bi_up, bo_right, bo_up, brd_color);
		ui->render_box(bi_left, bi_down, bo_left, bo_down, bo_right, bo_down, bi_right, bi_down, brd_color);
		ui->render_box(bi_right, bi_up, bi_right, bi_down, bo_right, bo_down, bo_right, bo_up, brd_color);

		{
			t_left = left + text_alignment.left - env.v[0];
			t_down = down + 4 + text_alignment.down - env.v[1];

			float dy = 0;
			float scale = 1.;

			float xpos = 0;
			float ypos = 0;

			float m_width = b_right - b_left - text_alignment.left - text_alignment.right;
			float m_height = b_up - b_down - text_alignment.up - text_alignment.down;
			float d_width = 0;
			float d_height = m_height - (env.v[3] - env.v[1]) - 4;

			std::vector<std::string> lines;
			std::string line;
			for (char c : text) {
				if (c == '\n') {
					d_width = m_width - ui->envelope_line(line, scale);
					if (text_alignment_horizontal_type == WIDGET_TEXT_ALIGNMENT_RIGHT_H) xpos = d_width + t_left;
					if (text_alignment_horizontal_type == WIDGET_TEXT_ALIGNMENT_LEFT_H) xpos = t_left;
					if (text_alignment_horizontal_type == WIDGET_TEXT_ALIGNMENT_CENTER_H)  xpos = 0.5*d_width + t_left;
					if (text_alignment_vertical_type == WIDGET_TEXT_ALIGNMENT_TOP_V) ypos = t_down + dy;
					if (text_alignment_vertical_type == WIDGET_TEXT_ALIGNMENT_BOTTOM_V) ypos = t_down + d_height + dy;
					if (text_alignment_vertical_type == WIDGET_TEXT_ALIGNMENT_CENTER_V)  ypos = t_down + 0.5*d_height + dy;
					ui->render_line(line, xpos, ypos, scale, txt_color);
					dy -= 14 * scale;
					line.clear();
					continue;
				}
				line.push_back(c);
			}
			d_width = m_width - ui->envelope_line(line, scale);
			if (text_alignment_horizontal_type == WIDGET_TEXT_ALIGNMENT_RIGHT_H) xpos = d_width + t_left;
			if (text_alignment_horizontal_type == WIDGET_TEXT_ALIGNMENT_LEFT_H) xpos = t_left;
			if (text_alignment_horizontal_type == WIDGET_TEXT_ALIGNMENT_CENTER_H)  xpos = 0.5*d_width + t_left;
			if (text_alignment_vertical_type == WIDGET_TEXT_ALIGNMENT_TOP_V) ypos = t_down + dy;
			if (text_alignment_vertical_type == WIDGET_TEXT_ALIGNMENT_BOTTOM_V) ypos = t_down + d_height + dy;
			if (text_alignment_vertical_type == WIDGET_TEXT_ALIGNMENT_CENTER_V)  ypos = t_down + 0.5*d_height + dy;
			ui->render_line(line, xpos, ypos, scale, txt_color);
		}

		return return_state;
	}

	void wrap_text(UI* ui) {
		float left = pos.left;
		float right = pos.right;
		float up = pos.up;
		float down = pos.down;

		vec4 env = ui->envelope_text(text, 1.0);

		pos.right = pos.left + text_alignment.left + env.v[2] - env.v[0] + text_alignment.right;
		pos.up = pos.down + text_alignment.down + env.v[3] - env.v[1] + 4 + text_alignment.up;
	}

};


int app_resize = 0;
void app_size_callback(GLFWwindow* window, int height, int width) {
	app_resize = 1;
}

struct body {
	struct {
		float x, y, z;
	} p, v, p_n, v_n;
	float r, m;
};

float h = 0.03;
float time = 0;

enum UI_CODE {
	UI_CODE_NONE = 0,
	UI_CODE_EXIT = 1,
};

class App {
	GLFWwindow* window;
	float width;
	float height;
	float aspect_ratio;

	UI ui;

	std::vector<body> bodies;

	void init_bodies() {
		// här specifieras storlek/massor/initellt läge för olika planeter
		bodies.push_back({ 0, 0, 0,			0, 0, 0,		0., 0., 0., 0., 0., 0.,		10.9, 3330 });
		bodies.push_back({ -3, -23.46, 0,	-13, 0, 0,		0., 0., 0., 0., 0., 0.,		1, 0.01 });
		bodies.push_back({ 2, 36.5, 0,		11, 0, 0,		0., 0., 0., 0., 0., 0.,		2, 0.08 });
		bodies.push_back({ 5, -45, 0,		-10, 0, 0,		0., 0., 0., 0., 0., 0.,		3, 0.27 });
		bodies.push_back({ 5, 60, 0,		8, 0, 0,		0., 0., 0., 0., 0., 0.,		5, 1.25 });
		bodies.push_back({ -3, -95, 0,		-5, 0, 0,		0., 0., 0., 0., 0., 0.,		2, 0.08 });

	}

	void body_vel_int(struct body* b1, struct body* b2) {
		struct {
			float x, y, z;
		} s1, s2, s3, s4, p;
		float Q;

		// RK4

		p.x = b1->p.x;
		p.y = b1->p.y;
		p.z = b1->p.z;
		Q = b2->m * powf(0.001 + (p.x - b2->p.x)*(p.x - b2->p.x) + (p.y - b2->p.y)*(p.y - b2->p.y) + (p.z - b2->p.z)*(p.z - b2->p.z), -1.5);
		s1.x = Q * (b2->p.x - p.x);
		s1.y = Q * (b2->p.y - p.y);
		s1.z = Q * (b2->p.z - p.z);

		p.x = b1->p.x + s1.x*h/2;
		p.y = b1->p.y + s1.y*h/2;
		p.z = b1->p.z + s1.z*h/2;
		Q = b2->m * powf(0.001 + (p.x - b2->p.x)*(p.x - b2->p.x) + (p.y - b2->p.y)*(p.y - b2->p.y) + (p.z - b2->p.z)*(p.z - b2->p.z), -1.5);
		s2.x = Q * (b2->p.x - p.x);
		s2.y = Q * (b2->p.y - p.y);
		s2.z = Q * (b2->p.z - p.z);

		p.x = b1->p.x + s2.x*h/2;
		p.y = b1->p.y + s2.y*h/2;
		p.z = b1->p.z + s2.z*h/2;
		Q = b2->m * powf(0.001 + (p.x - b2->p.x)*(p.x - b2->p.x) + (p.y - b2->p.y)*(p.y - b2->p.y) + (p.z - b2->p.z)*(p.z - b2->p.z), -1.5);
		s3.x = Q * (b2->p.x - p.x);
		s3.y = Q * (b2->p.y - p.y);
		s3.z = Q * (b2->p.z - p.z);

		p.x = b1->p.x + s3.x*h;
		p.y = b1->p.y + s3.y*h;
		p.z = b1->p.z + s3.z*h;
		Q = b2->m * powf(0.001 + (p.x - b2->p.x)*(p.x - b2->p.x) + (p.y - b2->p.y)*(p.y - b2->p.y) + (p.z - b2->p.z)*(p.z - b2->p.z), -1.5);
		s4.x = Q * (b2->p.x - p.x);
		s4.y = Q * (b2->p.y - p.y);
		s4.z = Q * (b2->p.z - p.z);

		b1->v_n.x += h / 6. * (s1.x + 2. * s2.x + 2. * s3.x + s4.x);
		b1->v_n.y += h / 6. * (s1.y + 2. * s2.y + 2. * s3.y + s4.y);
		b1->v_n.z += h / 6. * (s1.z + 2. * s2.z + 2. * s3.z + s4.z);

	}
	
	void body_pos_int(struct body* b1) {
		b1->p_n.x += h * (b1->v.x);
		b1->p_n.y += h * (b1->v.y);
		b1->p_n.z += h * (b1->v.z);
	}

	void update_bodies() {
		int i, j, n = bodies.size();
		for (i = 0; i < n; i++) {
			body_pos_int(&bodies[i]);

			for (j = 0; j < n; j++) {
				if (i == j) continue;
				body_vel_int(&bodies[i], &bodies[j]);

			}
		}

		for (i = 0; i < n; i++) {
			struct body* body = &bodies[i];
			body->v.x += body->v_n.x;
			body->v.y += body->v_n.y;
			body->v.z += body->v_n.z;
			body->v_n.x = 0.;
			body->v_n.y = 0.;
			body->v_n.z = 0.;

			body->p.x += body->p_n.x;
			body->p.y += body->p_n.y;
			body->p.z += body->p_n.z;
			body->p_n.x = 0.;
			body->p_n.y = 0.;
			body->p_n.z = 0.;
		}
	}

	void draw_bodies(vec4 light, float pitch, float yaw, float x, float y, float z, planet* P) {
		int i;
		for (i = 0; i < bodies.size(); i++) {
			auto body = bodies[i];
			if (i == 0) P->draw_sun(time, pitch, yaw, x + body.p.x, y + body.p.y, z + body.p.z, body.r);
			else {
				P->draw_planet(light, time, pitch, yaw, x + body.p.x, y + body.p.y, z + body.p.z, body.r);
			}
		}
	}

	int fullscreen_on = 0;

	Widget theme, sadd_theme, saddf_theme, dd_theme;
	Widget exited, floating, minimized, title;
	Widget file, edit, view, help;
	Widget file_s, edit_s, view_s, help_s;
	Widget file_open, file_open_new, file_open_recent, file_save, file_export;
	Widget tooltip, cursor;
	void UI_setup() {
		glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
		//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

		theme.set_border(0, 0., 0, 0.);
		theme.set_pos(0, 200, 0, 200);
		theme.set_alignment(7, 7, 2, 2);
		theme.text = std::string("the");
		theme.alignment_horizontal_type = WIDGET_ALIGNMENT_RIGHT_H;
		theme.alignment_vertical_type = WIDGET_ALIGNMENT_TOP_V;
		theme.border_alignment_type = WIDGET_BORDER_INSIDE;
		theme.text_alignment_horizontal_type = WIDGET_TEXT_ALIGNMENT_CENTER_H;
		theme.text_alignment_vertical_type = WIDGET_TEXT_ALIGNMENT_CENTER_V;
		theme.wrap_text(&ui);
		
		// PRIO 1: window functions
		exited = theme;
		exited.text = std::string("X");
		exited.wrap_text(&ui);
		floating = theme;
		floating.text = std::string("U");
		floating.wrap_text(&ui);
		floating.set_beside_widget(&exited);
		minimized = theme;
		minimized.text = std::string("_");
		minimized.wrap_text(&ui);
		minimized.set_beside_widget(&floating);
		title = theme;
		
		char debug_buffer[256];
		sprintf(debug_buffer, "mspf: %f", mspf);

		title.text = std::string(debug_buffer);
		title.wrap_text(&ui);
		title.alignment_horizontal_type = WIDGET_ALIGNMENT_LEFT_H;
		title.text_alignment_horizontal_type = WIDGET_TEXT_ALIGNMENT_LEFT_H;
		title.state_type = WIDGET_STATE_LOCKED;

		// Prio 2: self-aligning drop-down
		sadd_theme.set_border(0, 0., 0, 0.);
		sadd_theme.set_pos(0, 200, 0, 200);
		sadd_theme.set_alignment(10, 10, 2, 2);
		sadd_theme.text = std::string("the");
		sadd_theme.alignment_horizontal_type = WIDGET_ALIGNMENT_LEFT_H;
		sadd_theme.alignment_vertical_type = WIDGET_ALIGNMENT_TOP_V;
		sadd_theme.border_alignment_type = WIDGET_BORDER_INSIDE;
		sadd_theme.text_alignment_horizontal_type = WIDGET_TEXT_ALIGNMENT_CENTER_H;
		sadd_theme.text_alignment_vertical_type = WIDGET_TEXT_ALIGNMENT_CENTER_V;
		sadd_theme.wrap_text(&ui);

		saddf_theme.set_border(0, 0., 0, 0.);
		saddf_theme.set_pos(0, 200, 0, 200);
		saddf_theme.set_alignment(10, 10, 2, 2);
		saddf_theme.text = std::string("the");
		saddf_theme.alignment_horizontal_type = WIDGET_ALIGNMENT_LEFT_H;
		saddf_theme.alignment_vertical_type = WIDGET_ALIGNMENT_TOP_V;
		saddf_theme.border_alignment_type = WIDGET_BORDER_INSIDE;
		saddf_theme.text_alignment_horizontal_type = WIDGET_TEXT_ALIGNMENT_CENTER_H;
		saddf_theme.text_alignment_vertical_type = WIDGET_TEXT_ALIGNMENT_CENTER_V;
		saddf_theme.state_type = WIDGET_STATE_LOCKED;
		saddf_theme.background_color.locked = vec4{0.1, 0.1, 0.1, 1.0};
		saddf_theme.wrap_text(&ui);

		// Prio 3: typisk dropdown
		dd_theme.set_border(0, 0., 0, 0.);
		dd_theme.set_pos(0, 200, 0, 20);
		dd_theme.set_alignment(10, 10, 2, 2);
		dd_theme.text = std::string("the");
		dd_theme.alignment_horizontal_type = WIDGET_ALIGNMENT_LEFT_H;
		dd_theme.alignment_vertical_type = WIDGET_ALIGNMENT_TOP_V;
		dd_theme.border_alignment_type = WIDGET_BORDER_INSIDE;
		dd_theme.text_alignment_horizontal_type = WIDGET_TEXT_ALIGNMENT_LEFT_H;
		dd_theme.text_alignment_vertical_type = WIDGET_TEXT_ALIGNMENT_CENTER_V;
		dd_theme.state_type = WIDGET_STATE_UNLOCKED;
		dd_theme.background_color.passive = saddf_theme.background_color.locked;


		{
			file = sadd_theme;
			file.text = std::string("file");
			file.wrap_text(&ui);
			if (!fullscreen_on) file.set_below_widget(&title);

			file_s = saddf_theme;
			file_s.text = std::string("file");
			file_s.wrap_text(&ui);
			if (!fullscreen_on) file_s.set_below_widget(&title);

			file_open = dd_theme;
			file_open.text = std::string("open");
			file_open.set_below_widget(&file_s);

			file_open_new = dd_theme;
			file_open_new.text = std::string("open new");
			file_open_new.set_beside_widget(&file_open);

			file_open_recent = dd_theme;
			file_open_recent.text = std::string("open_recent");
			file_open_recent.set_below_widget(&file_open_new);

			file_save = dd_theme;
			file_save.text = std::string("save");
			file_save.set_below_widget(&file_open);

			file_export = dd_theme;
			file_export.text = std::string("export");
			file_export.set_below_widget(&file_save);
		}

		{
			edit = sadd_theme;
			edit.text = std::string("edit");
			edit.wrap_text(&ui);
			edit.set_beside_widget(&file);
			if (edit.pos.right > width) edit.set_below_widget(&file);

			edit_s = saddf_theme;
			edit_s.text = std::string("edit");
			edit_s.wrap_text(&ui);
			edit_s.set_beside_widget(&file);
			if (edit_s.pos.right > width) edit_s.set_below_widget(&file);
			
		}

		{
			view = sadd_theme;
			view.text = std::string("view");
			view.wrap_text(&ui);
			view.set_beside_widget(&edit);
			if (view.pos.right > width) view.set_below_widget(&edit);

			view_s = saddf_theme;
			view_s.text = std::string("view");
			view_s.wrap_text(&ui);
			view_s.set_beside_widget(&edit);
			if (view_s.pos.right > width) view_s.set_below_widget(&edit);

		}

		{
			help = sadd_theme;
			help.text = std::string("help");
			help.wrap_text(&ui);
			help.set_beside_widget(&view);
			if (help.pos.right > width) help.set_below_widget(&view);

			help_s = saddf_theme;
			help_s.text = std::string("help");
			help_s.wrap_text(&ui);
			help_s.set_beside_widget(&view);
			if (help_s.pos.right > width) help_s.set_below_widget(&view);
			
		}
		// Tooltip, cursor

		tooltip = theme;
		tooltip.text = std::string("Basic tooltip\nWherein text is and\nevery algorithm works as intended.");
		tooltip.set_border(1, 1, 1, 1);
		tooltip.background_color.passive = vec4{ 0.2, 0.2, 0.2, 0.6 };
		tooltip.alignment_horizontal_type = WIDGET_ALIGNMENT_LEFT_H;
		tooltip.text_alignment_horizontal_type = WIDGET_TEXT_ALIGNMENT_LEFT_H;
		tooltip.state_type = WIDGET_STATE_FLIPPABLE;
		tooltip.wrap_text(&ui);


		cursor = theme;
		cursor.text = std::string("^");
		cursor.set_alignment(-4., 0., 0., 0.);
		cursor.background_color.passive = vec4{ 0.2, 0.2, 0.2, 0.0 };
		cursor.alignment_horizontal_type = WIDGET_ALIGNMENT_LEFT_H;
		cursor.text_alignment_horizontal_type = WIDGET_TEXT_ALIGNMENT_LEFT_H;
		cursor.text_alignment_vertical_type = WIDGET_TEXT_ALIGNMENT_TOP_V;
		cursor.state_type = WIDGET_STATE_UNLOCKED;
		cursor.wrap_text(&ui);
	}

	int double_click_bit = -1;
	int floating_mode = 0;
	float screen_x_drag2 = 0.;
	float screen_y_drag2 = 0.;
	float screen_x_drag1 = 0.;
	float screen_y_drag1 = 0.;
	int pos_x_drag1 = 0.;
	int pos_y_drag1 = 0.;
	int pos_x_drag2 = 0.;
	int pos_y_drag2 = 0.;
	int mouse_drag0 = 0;
	int mouse_drag1 = 0;
	int mouse_drag2 = 0;
	float timer0 = 0;
	float timer_bit = 0;

	int menu_activated = 0;
	int menu_file_activated = 0;
	
	int in_hit_box(float x, float y, float left, float right, float up, float down) {
		return (left < x) & (x < right) & (down < (height - y)) & ((height - y) < up);
	}

	UI_CODE UI_loop() {
		double mouse_x = 0.;
		double mouse_y = 0.;
		glfwGetCursorPos(window, &mouse_x, &mouse_y);
		uint32_t lclick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
		uint32_t rclick = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

		if (fullscreen_on) {
			ui.render_box(0, height, 0, height - help.pos.up, width, height - help.pos.up, width, height, theme.background_color.passive);

			mouse_drag2 = mouse_drag1;
		}
		else {
			ui.render_box(0, height, 0, height - help.pos.up, width, height - help.pos.up, width, height, theme.background_color.passive);

			title.draw_widget(&ui, 0, 0, 0, 0);
			if (exited.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y) == WIDGET_RETURN_STATE_RELEASED) return UI_CODE_EXIT;
			if (floating.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y) == WIDGET_RETURN_STATE_RELEASED) {
				if ((++floating_mode % 2)) {
					glfwMaximizeWindow(window);
				}
				else {
					glfwRestoreWindow(window);
				}
			}
			if (minimized.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y) == WIDGET_RETURN_STATE_RELEASED) glfwIconifyWindow(window);

			mouse_drag2 = mouse_drag1;

			in_hit_box(mouse_x, mouse_y, 2, width - 2, height - 2, height - theme.pos.up - 2); // dragger
			in_hit_box(mouse_x, mouse_y, 0, width, height, height - 2); // top
			in_hit_box(mouse_x, mouse_y, 0, 2, height, 0); // left
			in_hit_box(mouse_x, mouse_y, width - 2, width, height, 0); // right
			in_hit_box(mouse_x, mouse_y, 0, width, 2, 0); // bottom
			in_hit_box(mouse_x, mouse_y, 2, width - 2, height - theme.pos.up, 2); // none

			if ((!in_hit_box(mouse_x, mouse_y, 2, width - 2, height - theme.pos.up, 2) & lclick) | ((mouse_drag0 != 0) & lclick)) {
				if (mouse_drag1 == 0) {
					mouse_drag0 = 0;
					glfwGetWindowPos(window, &pos_x_drag1, &pos_y_drag1);
					screen_x_drag1 = pos_x_drag1 + mouse_x;
					screen_y_drag1 = pos_y_drag1 + mouse_y;
					mouse_drag1 = 1;

					if (timer_bit == 0) {
						timer0 = glfwGetTime();
						timer_bit = 1;
					}
					else {
						if (glfwGetTime() - timer0 < 0.22) {
							if ((++floating_mode % 2)) {
								glfwMaximizeWindow(window);
							}
							else {
								glfwRestoreWindow(window);
							}
						}
						timer_bit = 0;
					}
				}

				if (mouse_drag0 == 0) {
					if (in_hit_box(mouse_x, mouse_y, 0, width, height - 2, height - theme.pos.up - 2))	mouse_drag0 = 1; // dragger
					if (in_hit_box(mouse_x, mouse_y, 0, width, height, height - 2))						mouse_drag0 |= 2; // top
					if (in_hit_box(mouse_x, mouse_y, 0, 2, height, 0))									mouse_drag0 |= 4; // left
					if (in_hit_box(mouse_x, mouse_y, width - 2, width, height, 0))						mouse_drag0 |= 8; // right
					if (in_hit_box(mouse_x, mouse_y, 0, width, 2, 0))									mouse_drag0 |= 16; // bottom
				}

				if ((mouse_drag1 == 1) & (mouse_drag2 == 1) & (mouse_drag0 & 1)) {
					glfwGetWindowPos(window, &pos_x_drag2, &pos_y_drag2);
					screen_x_drag2 = pos_x_drag2 + mouse_x;
					screen_y_drag2 = pos_y_drag2 + mouse_y;
					glfwSetWindowPos(window, pos_x_drag1 - screen_x_drag1 + screen_x_drag2, pos_y_drag1 - screen_y_drag1 + screen_y_drag2);
				}
				if(!(floating_mode % 2)){
					if ((mouse_drag1 == 1) & (mouse_drag2 == 1) & (0 < (mouse_drag0 & 8))) {
						glfwGetWindowPos(window, &pos_x_drag2, &pos_y_drag2);
						screen_x_drag2 = pos_x_drag2 + mouse_x;
						screen_y_drag2 = pos_y_drag2 + mouse_y;
						glfwSetWindowSize(window, mouse_x, height);
					}
					if ((mouse_drag1 == 1) & (mouse_drag2 == 1) & (0 < (mouse_drag0 & 16))) {
						glfwGetWindowPos(window, &pos_x_drag2, &pos_y_drag2);
						screen_x_drag2 = pos_x_drag2 + mouse_x;
						screen_y_drag2 = pos_y_drag2 + mouse_y;
						glfwSetWindowSize(window, width, mouse_y);
					}
				}

			}
			else {
				mouse_drag1 = 0;
				mouse_drag0 = 0;
			}
			if (glfwGetTime() - timer0 > 0.22) {
				timer_bit = 0;
			}
		}


		// det roliga kommer nedan :PPPPP fuck vilken clusterfuck av skit :SDASDPAOp
		
		WIDGET_RETURN_STATE file_state = file.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);
		WIDGET_RETURN_STATE edit_state = edit.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);
		WIDGET_RETURN_STATE view_state = view.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);
		WIDGET_RETURN_STATE help_state = help.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);

		if (menu_activated == 0) {
			if (file_state == WIDGET_RETURN_STATE_PRESSED) menu_activated = 1;
			if (edit_state == WIDGET_RETURN_STATE_PRESSED) menu_activated = 2;
			if (view_state == WIDGET_RETURN_STATE_PRESSED) menu_activated = 3;
			if (help_state == WIDGET_RETURN_STATE_PRESSED) menu_activated = 4;
		}
		else {
			if ((menu_activated == 1) & (file_state == WIDGET_RETURN_STATE_PRESSED)) menu_activated = 0;
			if (file_state == WIDGET_RETURN_STATE_HOVER) menu_activated = 1;
			if ((menu_activated == 2) & (edit_state == WIDGET_RETURN_STATE_PRESSED)) menu_activated = 0;
			if (edit_state == WIDGET_RETURN_STATE_HOVER) menu_activated = 2;
			if ((menu_activated == 3) & (view_state == WIDGET_RETURN_STATE_PRESSED)) menu_activated = 0;
			if (view_state == WIDGET_RETURN_STATE_HOVER) menu_activated = 3;
			if ((menu_activated == 4) & (help_state == WIDGET_RETURN_STATE_PRESSED)) menu_activated = 0;
			if (help_state == WIDGET_RETURN_STATE_HOVER) menu_activated = 4;
		}

		if (menu_activated == 1) {
			file_s.draw_widget(&ui, 0, 0, 0, 0);
			WIDGET_RETURN_STATE file_open_state = file_open.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);
			WIDGET_RETURN_STATE file_save_state = file_save.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);

			if (menu_file_activated == 0) {
				if (file_open_state == WIDGET_RETURN_STATE_HOVER) menu_file_activated = 1;
			}
			else {
				if (file_open_state == WIDGET_RETURN_STATE_HOVER) menu_file_activated = 1;
				if (file_save_state == WIDGET_RETURN_STATE_HOVER) menu_file_activated = 0;
			}

			if (menu_file_activated == 1) {
				WIDGET_RETURN_STATE file_open_new_state = file_open_new.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);
				WIDGET_RETURN_STATE file_open_recent_state = file_open_recent.draw_widget(&ui, lclick, rclick, mouse_x, mouse_y);

			}

		}
		else menu_file_activated = 0;

		if (menu_activated == 2) {
			edit_s.draw_widget(&ui, 0, 0, 0, 0);

		}
		if (menu_activated == 3) {
			view_s.draw_widget(&ui, 0, 0, 0, 0);

		}
		if (menu_activated == 4) {
			help_s.draw_widget(&ui, 0, 0, 0, 0);

		}


		

		//tooltip.set_pos(mouse_x, 0, mouse_y, 0);
		//tooltip.text = std::string("Basic tooltip\nWherein text is and\nevery algorithm works as intended.");
		//tooltip.wrap_text(&ui);
		//tooltip.draw_widget(&ui, 0, 0, 0, 0);
		

		//cursor.set_pos(mouse_x, 0, mouse_y - 4, 0);
		//cursor.text = std::string("^");
		//cursor.wrap_text(&ui);
		//cursor.draw_widget(&ui, 0, 0, 0, 0);


		return UI_CODE_NONE;
	}


	void size_fun() {
		int _width, _height;
		glfwGetWindowSize(window, &_width, &_height);
		width = _width;
		height = _height;
		glViewport(0, 0, width, height);
	}


public:
	App() : window(NULL), width(1920 * 0.6), height(1080 * 0.6), ui(&width, &height), aspect_ratio((float)width/(float)height) {
		glfwInit();

		window = glfwCreateWindow(width, height, "Title", NULL, NULL);

		glfwMakeContextCurrent(window);

		glewInit();

		FreeImage_Initialise();

		ui.initialize();

	}

	~App() {
		glfwTerminate();

		FreeImage_DeInitialise();
	}
	
	double lastTime = glfwGetTime();
	int nbFrames = 0;
	float mspf = 0;

	int main() {

		glfwSetWindowSizeCallback(window, &app_size_callback);

		planet P;

		init_bodies();

		glfwSwapInterval(1);

		glEnable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);

		float zoom = -80.;


		float x = 0.;
		float y = 0.;
		
		float pitch = 0.;
		float yaw = 0.;

		mat4 mv;
		vec4 light;

		UI_setup();
		int UI_SETUP_RATE = 100;
		int UI_SETUP_RATEv = 0;

		double mouse_x = 0.;
		double mouse_y = 0.;

		while (!glfwWindowShouldClose(window)) {
			{
				double currentTime = glfwGetTime();
				nbFrames++;
				if (currentTime - lastTime >= 0.1) { // If last prinf() was more than 1 sec ago
					// printf and reset timer
					mspf = 100.0 / double(nbFrames);
					nbFrames = 0;
					lastTime += 0.1;
				}
			}
			if (app_resize) size_fun();

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glClearColor(0.1, 0.1, 0.16, 1.0);

			time = fmodf(time + 0.01, 314.159265359);

			update_bodies();

			mat4 R = matmat_mul(pitch_matrix(pitch), roll_matrix(yaw));
			mat4 mv = matmat_mul(R, translation_matrix(x, y, zoom));
			light = matvec_mul(mv, vec4{ 0., 0., 0., 1. });


			draw_bodies(light, pitch, yaw, x, y, zoom, &P);
			
			if (!(UI_SETUP_RATEv++ % UI_SETUP_RATE)) UI_setup();
			if (UI_loop() == UI_CODE_EXIT) break;


			glfwGetCursorPos(window, &mouse_x, &mouse_y);
			if (glfwGetKey(window, GLFW_KEY_ESCAPE)) break;
			if (glfwGetKey(window, GLFW_KEY_W)) { x -= sin(yaw); y -= cos(yaw); };
			if (glfwGetKey(window, GLFW_KEY_S)) { x += sin(yaw); y += cos(yaw); };
			if (glfwGetKey(window, GLFW_KEY_A)) { x += cos(yaw); y -= sin(yaw); };
			if (glfwGetKey(window, GLFW_KEY_D)) { x -= cos(yaw); y += sin(yaw); };
			if (glfwGetKey(window, GLFW_KEY_UP))    pitch -= .01;
			if (glfwGetKey(window, GLFW_KEY_DOWN))  pitch += .01;
			if (glfwGetKey(window, GLFW_KEY_LEFT))  yaw -= .02;
			if (glfwGetKey(window, GLFW_KEY_RIGHT)) yaw += .02;
			if (glfwGetKey(window, GLFW_KEY_Q))  yaw -= .02;
			if (glfwGetKey(window, GLFW_KEY_E)) yaw += .02;
			if (glfwGetKey(window, GLFW_KEY_SPACE)) { x = 0, y = 0, zoom = -50.; };
			if (glfwGetKey(window, GLFW_KEY_Z)) zoom -= 1.;
			if (glfwGetKey(window, GLFW_KEY_X)) zoom += 1.;
			glfwPollEvents();
			glfwSwapBuffers(window);
		}

		return 1;
	}

};


int main() {
	App app;

	return app.main();
}