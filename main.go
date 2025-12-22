package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
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
	fs := flag.NewFlagSet("test", flag.ExitOnError)
	full := fs.Bool("full", false, "run bench on all valid tree combinations")
	var gpus GPUSlice
	fs.Var(&gpus, "gpus", "comma-separated list of GPU devices")
	var widths TreeRange
	fs.Var(&widths, "width", "comma-separated list of tree widths")
	var depths TreeRange
	fs.Var(&depths, "Depth", "comma-separated list of tree depths")
	draftModel := fs.String("draft_model", "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", "the model to use for drafting")
	targetModel := fs.String("target_model", "meta-llama/Llama-3.1-8B-Instruct", "the model to use for verification")
	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatal(err)
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

	if err := fullRun(ctx, gpus, widths, depths, os.Environ(), *draftModel, *targetModel); err != nil {
		log.Fatal(err)
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

func fullRun(ctx context.Context, gpus []string, widths, depths []int, environ []string, draftModel, targetModel string) error {
	ch := generateWork(ctx, widths, depths)
	errsCh := startWorkers(ctx, ch, gpus, environ, draftModel, targetModel)
	var errs []error
	for err := range errsCh {
		if err != nil {
			log.Println(err)
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

func startWorkers(ctx context.Context, ch <-chan work, gpus []string, environ []string, draftModel, targetModel string) <-chan error {
	c := make(chan error, 1)
	var wg errgroup.Group
	port := 9000
	for _, gpu := range gpus {
		wg.Go(func(port int) func() error {
			return func() error {
				gpu := "CUDA_VISIBLE_DEVICES=" + gpu
				for w := range ch {
					log.Printf("worker %v: enableCudagraph=%v, width=%v, Depth=%v", port, w.enableCudagraph, w.width, w.depth)
					b := &BenchServeRunner{
						EnableCudagraph: w.enableCudagraph,
						Width:           w.width,
						Depth:           w.depth,
						Env:             append(environ, gpu),
						Port:            port,
						DraftModel:      draftModel,
						TargetModel:     targetModel,
					}
					if err := b.Setup(ctx, dag.NoParentConfig).Run(ctx); err != nil {
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
	g                       *dag.Graph
}

func (r *BenchServeRunner) Setup(ctx context.Context, configFunc dag.ParentConfigFunc) dag.Runner {
	g := dag.NewGraph()
	g.Add("serve",
		dag.WithRunner(&ServeConfig{
			env:             r.Env,
			targetModel:     r.TargetModel,
			port:            r.Port,
			enableCudagraph: r.EnableCudagraph,
			width:           r.Width,
			depth:           r.Depth,
			draftModel:      r.DraftModel,
		}),
	)
	g.Add("bench",
		dag.WithParents("serve"),
		dag.WithRunner(&BenchConfig{
			env: r.Env,
		}),
	)
	g.Add("file printer",
		dag.WithParents("serve", "bench"),
		dag.WithRunner(BenchServeOutFileRunner{}),
	)
	r.g = g
	return r
}

func (r *BenchServeRunner) Run(ctx context.Context) error {
	return r.g.Run(ctx)
}

func (r *BenchServeRunner) RunConfig(s string) any {
	return nil
}

type ServeConfig struct {
	env             []string
	targetModel     string
	port            int
	enableCudagraph bool
	width           int
	depth           int
	draftModel      string

	*cmd
}

func (s *ServeConfig) Setup(ctx context.Context, _ dag.ParentConfigFunc) dag.Runner {
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
	serveCtx, serveCancel := context.WithCancel(ctx)
	serveBuf := bytes.Buffer{}
	serveCmd := exec.CommandContext(serveCtx, serveArgs[0], serveArgs[1:]...)
	serveCmd.Env = myEnv
	serveCmd.Stdout = &serveBuf
	serveCmd.Stderr = &serveBuf
	serveCmd.WaitDelay = 10 * time.Second
	serveCmd.Cancel = func() error {
		err := serveCmd.Process.Signal(syscall.SIGTERM)
		if err != nil {
			return fmt.Errorf("failed to send SIGTERM to serveCmd: %w", err)
		}
		return nil
	}
	s.cmd = &cmd{
		cmd: serveCmd,
		config: func(key string) any {
			switch key {
			case "model":
				return s.targetModel
			case "port":
				return s.port
			case "buffer":
				return serveBuf
			case "cudagraph":
				return s.enableCudagraph
			default:
				return nil
			}
		},
		cancel: serveCancel,
	}
	return s
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
	parent string
	env    []string
	*cmd
}

func (b *BenchConfig) Setup(ctx context.Context, configFunc dag.ParentConfigFunc) dag.Runner {
	targetModel := configFunc(b.parent, "model").(string)
	port := configFunc(b.parent, "port").(int)
	benchArgs := []string{
		"vllm",
		"bench",
		"serve",
		"--model=" + targetModel,
		"--tokenizer=" + targetModel,
		"--dataset-name=hf",
		"--dataset-path=philschmid/mt-bench",
		"--ignore-eos",
		"--request-rate=inf",
		"--max-concurrency=1",
		"--num-prompts=80",
		fmt.Sprintf("--port=%d", port),
	}

	benchCtx, benchCancel := context.WithCancel(ctx)
	defer benchCancel()
	benchBuf := bytes.Buffer{}
	benchCmd := exec.CommandContext(benchCtx, benchArgs[0], benchArgs[1:]...)
	benchCmd.Env = b.env
	benchCmd.Stdout = &benchBuf
	benchCmd.Stderr = &benchBuf
	benchCmd.WaitDelay = 10 * time.Second
	benchCmd.Cancel = func() error {
		err := benchCmd.Process.Signal(syscall.SIGTERM)
		if err != nil {
			return fmt.Errorf("failed to send SIGTERM to benchCmd: %w", err)
		}
		return nil
	}
	b.cmd = &cmd{
		cmd: benchCmd,
		config: func(key string) any {
			switch key {
			case "model":
				return targetModel
			case "port":
				return port
			case "buffer":
				return benchBuf
			default:
				return nil
			}
		},
		cancel: benchCancel,
	}
	return b
}

type cmd struct {
	cmd    *exec.Cmd
	config func(string) any
	cancel context.CancelFunc
}

func (c *cmd) Run(ctx context.Context) error {
	defer c.cancel()
	return c.cmd.Run()
}

func (c *cmd) RunConfig(key string) any {
	return c.config(key)
}

type BenchServeOutFileRunner struct {
	serveParent, benchParent string
	width, depth             int
	cudagraph                bool
	benchBuf                 *bytes.Buffer
	serveBuf                 *bytes.Buffer
}

func (r BenchServeOutFileRunner) Setup(ctx context.Context, configFunc dag.ParentConfigFunc) dag.Runner {
	return BenchServeOutFileRunner{
		width:     configFunc(r.serveParent, "width").(int),
		depth:     configFunc(r.serveParent, "Depth").(int),
		cudagraph: configFunc(r.serveParent, "cudagraph").(bool),
		serveBuf:  configFunc(r.serveParent, "buffer").(*bytes.Buffer),
		benchBuf:  configFunc(r.benchParent, "buffer").(*bytes.Buffer),
	}
}

func (r BenchServeOutFileRunner) RunConfig(s string) any {
	panic("implement me")
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
		log.Printf("failed to write bench result: %v", err)
	}
	_, err = w.WriteString("\n\n")
	if err != nil {
		log.Printf("failed to write newlines: %v", err)
	}
	err = w.Flush()
	if err != nil {
		log.Printf("failed to flush bench result: %v", err)
	}
	_, err = w.ReadFrom(r.serveBuf)
	if err != nil {
		log.Printf("failed to write serve result: %v", err)
	}
	err = w.Flush()
	if err != nil {
		log.Printf("failed to flush serve result: %v", err)
	}
	err = f.Close()
	if err != nil {
		return fmt.Errorf("failed to close file: %w", err)
	}
	return nil
}
