#include <GL/glew.h>
#include <GLFW/glfw3.h>

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
		char file_path[32];
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

class App {
	GLFWwindow* window;
	uint32_t width;
	uint32_t height;
	float aspect_ratio;

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


	void size_fun() {
		int _width, _height;
		glfwGetWindowSize(window, &_width, &_height);
		width = _width;
		height = _height;
		glViewport(0, 0, width, height);
	}


public:
	App() : window(NULL), width(1920 * 0.6), height(1080 * 0.6), aspect_ratio((float)width/(float)height) {
		glfwInit();

		window = glfwCreateWindow(width, height, "Title", NULL, NULL);

		glfwMakeContextCurrent(window);

		glewInit();

	}

	~App() {
		glfwTerminate();
	}

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

		while (!glfwWindowShouldClose(window)) {
			if (app_resize) size_fun();

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glClearColor(0.13, 0.13, 0.135, 1.);

			time = fmodf(time + 0.01, 314.159265359);

			update_bodies();

			mat4 R = matmat_mul(pitch_matrix(pitch), roll_matrix(yaw));
			mat4 mv = matmat_mul(R, translation_matrix(x, y, zoom));
			light = matvec_mul(mv, vec4{ 0., 0., 0., 1. });

			draw_bodies(light, pitch, yaw, x, y, zoom, &P);

			if (glfwGetKey(window, GLFW_KEY_ESCAPE)) break;
			if (glfwGetKey(window, GLFW_KEY_W)) { x -= sin(yaw); y -= cos(yaw); };
			if (glfwGetKey(window, GLFW_KEY_S)) { x += sin(yaw); y += cos(yaw); };
			if (glfwGetKey(window, GLFW_KEY_A)) { x += cos(yaw); y -= sin(yaw); };
			if (glfwGetKey(window, GLFW_KEY_D)) { x -= cos(yaw); y += sin(yaw); };
			if (glfwGetKey(window, GLFW_KEY_UP))    pitch -= .01;
			if (glfwGetKey(window, GLFW_KEY_DOWN))  pitch += .01;
			if (glfwGetKey(window, GLFW_KEY_LEFT))  yaw -= .02;
			if (glfwGetKey(window, GLFW_KEY_RIGHT)) yaw += .02;
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