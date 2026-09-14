class Hi < Formula
  desc "Verification-first coding agent"
  homepage "https://github.com/PipeNetwork/hi"
  head "https://github.com/PipeNetwork/hi.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", "--locked", "--root", prefix, "--path", "crates/hi-cli"
    system "cargo", "install", "--locked", "--root", prefix, "--path", "crates/hi-sentinel"
  end

  test do
    output = shell_output("#{bin}/hi --help")
    assert_match "setup", output
    assert_match "doctor", output
    sentinel = shell_output("#{bin}/hi-sentinel --help")
    assert_match "Sentinel", sentinel
  end
end
