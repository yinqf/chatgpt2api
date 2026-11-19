package handler

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/synctest"

	"github.com/basketikun/chatgpt2api/internal/service"
	"github.com/gin-gonic/gin"
)

func TestImageRequestCapacityAfterStreamCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		h := NewAIHandler(nil, nil, 1)
		router := gin.New()
		h.RegisterRoutes(router.Group("/v1"))
		router.POST("/test-stream", h.limitImageRequests, func(c *gin.Context) {
			startSSE(c)
			<-c.Request.Context().Done()
		})

		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		stream := httptest.NewRecorder()
		go router.ServeHTTP(stream, httptest.NewRequestWithContext(ctx, http.MethodPost, "/test-stream", nil))
		synctest.Wait()
		if !stream.Flushed {
			t.Fatal("stream did not start")
		}

		paths := []string{"/v1/images/generations", "/v1/images/edits"}
		for _, path := range paths {
			response := httptest.NewRecorder()
			router.ServeHTTP(response, httptest.NewRequest(http.MethodPost, path, strings.NewReader("{")))
			if response.Code != http.StatusTooManyRequests || response.Header().Get("Retry-After") != "5" {
				t.Fatalf("%s: status %d, Retry-After %q, body %s", path, response.Code, response.Header().Get("Retry-After"), response.Body)
			}
			var payload struct {
				Error struct {
					Type string `json:"type"`
					Code string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
				t.Fatal(err)
			}
			if payload.Error.Code != "image_capacity_exceeded" || payload.Error.Type != "rate_limit_error" {
				t.Fatalf("%s: unexpected error response %s", path, response.Body)
			}
		}

		chat := httptest.NewRecorder()
		router.ServeHTTP(chat, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{")))
		if chat.Code != http.StatusBadRequest {
			t.Fatalf("chat validation was interrupted by image capacity: status %d, body %s", chat.Code, chat.Body)
		}

		cancel()
		synctest.Wait()
		for _, path := range paths {
			response := httptest.NewRecorder()
			router.ServeHTTP(response, httptest.NewRequest(http.MethodPost, path, strings.NewReader("{")))
			if response.Code != http.StatusBadRequest {
				t.Fatalf("%s: expected validation to run after slot release, got %d with %s", path, response.Code, response.Body)
			}
		}
	})
}

func TestParseImageRequestJSON(t *testing.T) {
	png := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("png"))
	jpeg := base64.StdEncoding.EncodeToString([]byte("jpeg"))
	body := map[string]any{"prompt": "merge", "images": []any{png, map[string]any{"b64_json": jpeg}, map[string]any{"image_url": map[string]any{"url": png}}}}
	request, err := parseJSONImageRequest(t, "/v1/images/edits", body)
	if err != nil {
		t.Fatal(err)
	}
	if request.Model != "" {
		t.Fatalf("got model %q", request.Model)
	}
	if len(request.Images) != 3 || string(request.Images[0]) != "png" || string(request.Images[1]) != "jpeg" || string(request.Images[2]) != "png" {
		t.Fatalf("got images %#v", request.Images)
	}
}

func TestParseImageRequestKeepsModel(t *testing.T) {
	request, err := parseJSONImageRequest(t, "/v1/images/edits", map[string]any{"model": "gpt-image-2.6", "prompt": "edit", "image": base64.StdEncoding.EncodeToString([]byte("image"))})
	if err != nil {
		t.Fatal(err)
	}
	if request.Model != "gpt-image-2.6" {
		t.Fatalf("got model %q", request.Model)
	}
}

func TestParseImageRequestMultipart(t *testing.T) {
	body := bytes.NewBuffer(nil)
	writer := multipart.NewWriter(body)
	if err := writer.WriteField("prompt", "merge"); err != nil {
		t.Fatal(err)
	}
	for _, file := range []struct{ field, name, data string }{{"image", "one.png", "one"}, {"image", "two.jpg", "two"}, {"image[]", "three.webp", "three"}} {
		part, err := writer.CreateFormFile(file.field, file.name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := part.Write([]byte(file.data)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	context := testContext(httptest.NewRequest(http.MethodPost, "/v1/images/edits", body))
	context.Request.Header.Set("Content-Type", writer.FormDataContentType())
	request, err := parseImageRequest(context)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Images) != 3 || string(request.Images[0]) != "one" || string(request.Images[1]) != "two" || string(request.Images[2]) != "three" {
		t.Fatalf("got images %#v", request.Images)
	}
}

func TestParseImageRequestValidation(t *testing.T) {
	image := base64.StdEncoding.EncodeToString([]byte("image"))
	for _, test := range []struct {
		name, path string
		body       map[string]any
	}{
		{"prompt", "/v1/images/generations", map[string]any{}},
		{"count", "/v1/images/generations", map[string]any{"prompt": "draw", "n": 5}},
		{"edit image", "/v1/images/edits", map[string]any{"prompt": "edit"}},
		{"invalid image", "/v1/images/edits", map[string]any{"prompt": "edit", "image": image + "!"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := parseJSONImageRequest(t, test.path, test.body); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func parseJSONImageRequest(t *testing.T, path string, body map[string]any) (request service.ImageRequest, err error) {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		return request, err
	}
	context := testContext(httptest.NewRequest(http.MethodPost, path, bytes.NewReader(data)))
	context.Request.Header.Set("Content-Type", "application/json")
	return parseImageRequest(context)
}

func testContext(request *http.Request) *gin.Context {
	context, _ := gin.CreateTestContext(httptest.NewRecorder())
	context.Request = request
	return context
}
