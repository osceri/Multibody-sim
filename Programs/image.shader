//??GL_VERTEX_SHADER
#version 450 core

layout(location = 0) in vec4 vertex;

out vec2 TexCoords;

uniform mat4 projection;

void main()
{
	gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
	TexCoords = vertex.zw;
}

//??GL_FRAGMENT_SHADER
#version 450 core

in vec2 TexCoords;
out vec4 color;

uniform sampler2D tex;

void main()
{
	color = texture(tex, TexCoords);
}