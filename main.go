package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"go.aponeill.com/dag"
	"golang.org/x/sync/errgroup"
)

type SpeculativeConfig struct {
	Model                string
	Method               string
	NumSpeculativeTokens int
	SpeculativeTokenTree string
}

func (s SpeculativeConfig) String() string {
	return fmt.Sprintf("--speculative-config={\"model\": %q, \"method\": %q, \"num_speculative_tokens\": %d, \"speculative_token_tree\": %q}", s.Model, s.Method, s.NumSpeculativeTokens, s.SpeculativeTokenTree)
}

type GPUSlice []string

func (g *GPUSlice) String() string {
	return fmt.Sprint(*g)
}

func (g *GPUSlice) Set(value string) error {
	parts := strings.Split(value, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		*g = append(*g, part)
	}
	return nil
}

type TreeRange []int

func (i *TreeRange) String() string {
	return fmt.Sprint(*i)
}

func (i *TreeRange) Set(value string) error {
	if value == "" {
		*i = append(*i, 3) // default to 3
		return nil
	} else if strings.ToLower(value) == "all" {
		for j := 1; j <= 64; j++ {
			*i = append(*i, j)
		}
		return nil
	}
	parts := strings.Split(value, ",")
	for _, part := range parts {
		part = strings.TrimSpace(part)
		val, err := strconv.Atoi(part)
		if err != nil {
			return fmt.Errorf("could not parse int: %w", err)
		}
		*i = append(*i, val)
	}
	return nil
}

func main() {
	handlerOpts := &slog.HandlerOptions{
		Level:     slog.LevelInfo,
		AddSource: true,
	}
	logger := slog.New(slog.NewTextHandler(os.Stdout, handlerOpts))
	slog.SetDefault(logger)
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	full := fs.Bool("full", false, "run bench on all valid tree combinations")
	var gpus GPUSlice
	fs.Var(&gpus, "gpus", "comma-separated list of GPU devices")
	var widths TreeRange
	fs.Var(&widths, "width", "comma-separated list of tree widths")
	var depths TreeRange
	fs.Var(&depths, "depth", "comma-separated list of tree depths")
	draftModel := fs.String("draft_model", "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "the model to use for drafting")
	targetModel := fs.String("target_model", "meta-llama/Llama-3.1-8B-Instruct", "the model to use for verification")
	if err := fs.Parse(os.Args[1:]); err != nil {
		slog.Error("could not parse flags", "error", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if *full {
		depths = depths[:0]
		widths = widths[:0]
		for i := 1; i <= 64; i++ {
			depths = append(depths, i)
			widths = append(widths, i)
		}
	}

	if err := fullRun(ctx, gpus, widths, depths, os.Environ(), *draftModel, *targetModel, logger); err != nil {
		slog.Error("could not run bench on all valid tree combinations", "error", err)
		os.Exit(1)
	}

	select {
	case <-ctx.Done():
		stop()
	default:
	}
}

type work struct {
	enableCudagraph bool
	width           int
	depth           int
}

func fullRun(ctx context.Context, gpus []string, widths, depths []int, environ []string, draftModel, targetModel string, logger *slog.Logger) error {
	ch := generateWork(ctx, widths, depths)
	errsCh := startWorkers(ctx, ch, gpus, environ, draftModel, targetModel, logger)
	var errs []error
	for err := range errsCh {
		if err != nil {
			slog.Info("startWorkers returned error", "error", err)
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func generateWork(ctx context.Context, widths, depths []int) <-chan work {
	out := make(chan work, 1)
	go func() {
		defer close(out)
		for _, d := range depths {
			if d == 64 {
				continue
			}
			for _, w := range widths {
				if w == 1 || w*d >= 64 {
					continue
				}
				select {
				case <-ctx.Done():
					return
				case out <- work{
					enableCudagraph: true,
					width:           w,
					depth:           d,
				}:
				}
			}
		}
	}()
	return out
}

func startWorkers(ctx context.Context, ch <-chan work, gpus []string, environ []string, draftModel, targetModel string, logger *slog.Logger) <-chan error {
	c := make(chan error, 1)
	var wg errgroup.Group
	port := 9000
	for _, gpu := range gpus {
		wg.Go(func(port int) func() error {
			return func() error {
				gpu := "CUDA_VISIBLE_DEVICES=" + gpu
				for w := range ch {
					logger.Info("startWorker", "port", port, "enableCudagraph", w.enableCudagraph, "width", w.width, "depth", w.depth)
					var serveBuf, benchBuf bytes.Buffer
					b := &BenchServeRunner{
						EnableCudagraph: w.enableCudagraph,
						Width:           w.width,
						Depth:           w.depth,
						Env:             append(environ, gpu),
						Port:            port,
						DraftModel:      draftModel,
						TargetModel:     targetModel,
						ServeBuf: &serveBuf,
						BenchBuf: &benchBuf,
					}
					g := dag.NewGraph(dag.WithSequential(true))
					g.Add("worker", dag.WithRunner(b.Setup()))
					g.Add(
						"file_writer", 
						dag.WithParents("worker"),
						 dag.WithRunner(BenchServeOutFileRunner{
							width: w.width,
							depth: w.depth,
							cudagraph: w.enableCudagraph,
							serveBuf: &serveBuf,
							benchBuf: &benchBuf,
						 }))
					if err := g.Run(ctx); err != nil {
						c <- fmt.Errorf("failed to run worker %+v: %w", b, err)
					}
				}
				return nil
			}
		}(port))
		port++
	}
	go func() {
		if err := wg.Wait(); err != nil {
			c <- fmt.Errorf("failed to wait for workers: %v", err)
		}
		close(c)
	}()
	return c
}

type BenchServeRunner struct {
	EnableCudagraph         bool
	Width, Depth            int
	Env                     []string
	Port                    int
	DraftModel, TargetModel string
	ServeBuf, BenchBuf      io.Writer
	logger *slog.Logger
}

func (r *BenchServeRunner) Setup() dag.Runner {
	g := dag.NewGraph(dag.WithLogger(r.logger))
	s := &ServeConfig{
		env:             r.Env,
		targetModel:     r.TargetModel,
		port:            r.Port,
		enableCudagraph: r.EnableCudagraph,
		width:           r.Width,
		depth:           r.Depth,
		draftModel:      r.DraftModel,
		buf:             r.ServeBuf,
	}
	g.Add("serve", dag.WithRunner(s.Setup()))
	b := &BenchConfig{
		env:         r.Env,
		targetModel: r.TargetModel,
		port:        r.Port,
		buf:         r.BenchBuf,
	}
	g.Add("bench",
		dag.WithParents("serve"),
		dag.WithRunner(b.Setup()),
	)
	return g
}

type ServeConfig struct {
	env             []string
	targetModel     string
	port            int
	enableCudagraph bool
	width           int
	depth           int
	draftModel      string
	buf             io.Writer
}

func (s *ServeConfig) Setup() dag.Runner {
	myEnv := make([]string, len(s.env))
	copy(myEnv, s.env)
	myEnv = append(myEnv, "VLLM_USE_V1=1")

	serveArgs := []string{
		"vllm",
		"serve",
		s.targetModel,
		"--disable-log-requests",
		"--tensor-parallel-size=1",
		"--max-num-seqs=64",
		"--max-model-len=32768",
		"--no-enable-prefix-caching",
		fmt.Sprintf("--port=%d", s.port),
	}
	if !s.enableCudagraph {
		myEnv = append(myEnv, "CUDA_LAUNCH_BLOCKING=1")
		serveArgs = append(serveArgs, "--enforce-eager")
	}
	tree := makeTreeString(s.width, s.depth)
	specConfig := SpeculativeConfig{
		Model:                s.draftModel,
		Method:               "eagle",
		NumSpeculativeTokens: s.width * s.depth,
		SpeculativeTokenTree: tree,
	}
	serveArgs = append(serveArgs, specConfig.String())

	return &CmdRunner{
		stdout:    s.buf,
		stderr:    s.buf,
		args:      serveArgs,
		env:       myEnv,
		waitDelay: 10 * time.Second,
		cancelFunc: func(c *exec.Cmd) func() error {
			return func() error {
				err := c.Process.Signal(syscall.SIGTERM)
				if err != nil {
					return fmt.Errorf("failed to send SIGTERM to serveCmd: %w", err)
				}
				return nil
			}
		},
	}
}

func makeTreeString(width int, depth int) string {
	var temp []string
	zeros := strings.Split(strings.Repeat("0", depth), "")
	for i := 0; i < depth; i++ {
		for j := 0; j < width; j++ {
			temp = append(temp, fmt.Sprintf("(%v,", j)+strings.Join(zeros[:i], ",")+")")
		}
	}
	out := strings.Join(temp, ",")
	out = "[" + out + "]"
	return out
}

type BenchConfig struct {
	targetModel string
	port        int
	env         []string
	buf         io.Writer
}

func (b *BenchConfig) Setup() dag.Runner {
	benchArgs := []string{
		"vllm",
		"bench",
		"serve",
		"--model=" + b.targetModel,
		"--tokenizer=" + b.targetModel,
		"--dataset-name=hf",
		"--dataset-path=philschmid/mt-bench",
		"--ignore-eos",
		"--request-rate=inf",
		"--max-concurrency=1",
		"--num-prompts=80",
		fmt.Sprintf("--port=%d", b.port),
	}

	return &CmdRunner{
		stdout:    b.buf,
		stderr:    b.buf,
		args:      benchArgs,
		env:       b.env,
		waitDelay: 10 * time.Second,
		cancelFunc: func(c *exec.Cmd) func() error {
			return func() error {
				err := c.Process.Signal(syscall.SIGTERM)
				if err != nil {
					return fmt.Errorf("failed to send SIGTERM to benchCmd: %w", err)
				}
				return nil
			}
		},
	}
}

type CmdRunnerCancelFunc func(*exec.Cmd) func() error

type CmdRunner struct {
	stdout     io.Writer
	stderr     io.Writer
	args       []string
	env        []string
	waitDelay  time.Duration
	cancelFunc CmdRunnerCancelFunc
}

func (c *CmdRunner) Run(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, c.args[0], c.args[1:]...)
	cmd.Env = c.env
	cmd.Stdout = c.stdout
	cmd.Stderr = c.stderr
	cmd.WaitDelay = c.waitDelay
	cmd.Cancel = c.cancelFunc(cmd)
	return cmd.Run()
}

type BenchServeOutFileRunner struct {
	width     int
	depth     int
	cudagraph bool
	benchBuf  io.ReadWriter
	serveBuf  io.ReadWriter
}

func (r BenchServeOutFileRunner) Run(ctx context.Context) error {
	cudagraphLabel := "cudagraph"
	if !r.cudagraph {
		cudagraphLabel = "no-cudagraph"
	}
	outputPath := fmt.Sprintf("vllm_server.%dx%d.%s.log", r.width, r.depth, cudagraphLabel)
	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	w := bufio.NewWriter(f)
	_, err = w.ReadFrom(r.benchBuf)
	if err != nil {
		slog.Info("failed to write bench result", "error", err)
	}
	_, err = w.WriteString("\n\n")
	if err != nil {
		slog.Info("failed to write newlines", "error", err)
	}
	err = w.Flush()
	if err != nil {
		slog.Info("failed to flush bench result", "error", err)
	}
	_, err = w.ReadFrom(r.serveBuf)
	if err != nil {
		slog.Info("failed to write serve result", "error", err)
	}
	err = w.Flush()
	if err != nil {
		slog.Info("failed to flush serve result", "error", err)
	}
	err = f.Close()
	if err != nil {
		return fmt.Errorf("failed to close file: %w", err)
	}
	return nil
}
