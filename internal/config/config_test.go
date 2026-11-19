package config

import (
	"path/filepath"
	"testing"
)

func TestImageMaxWorkersFromEnvironment(t *testing.T) {
	t.Setenv("CHATGPT2API_AUTH_KEY", "test")
	path := filepath.Join(t.TempDir(), "config.yaml")
	for _, test := range []struct {
		name  string
		value string
		want  int
	}{
		{name: "default", want: 100},
		{name: "invalid", value: "many", want: 100},
		{name: "configured", value: " 8 ", want: 8},
		{name: "zero", value: "0", want: 1},
		{name: "negative", value: "-3", want: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("CHATGPT2API_IMAGE_MAX_WORKERS", test.value)
			cfg, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if cfg.ImageMaxWorkers != test.want {
				t.Fatalf("image max workers = %d, want %d", cfg.ImageMaxWorkers, test.want)
			}
		})
	}
}
