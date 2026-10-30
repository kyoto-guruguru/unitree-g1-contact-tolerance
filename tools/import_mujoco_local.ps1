# Run from the root of the extracted staging repository on the Windows PC.
# SourceRoot is read-only. This script does not push to GitHub.
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Source directory does not exist: $SourceRoot"
}
$copyList = @(
    @('G1_P012_MuJoCo_Windows_Package\run_p012_windows.py','mujoco\scripts\run_p012_windows.py'),
    @('scripts\record_g1.py','mujoco\scripts\record_g1.py'),
    @('results\videos\MuJoCo_P012_comparison\edit_p012_3panel.py','mujoco\scripts\edit_p012_3panel.py'),
    @('G1_P012_MuJoCo_Windows_Package\deploy\deploy_mujoco\configs\g1_P0_6700.yaml','mujoco\configs\g1_P0_6700.yaml'),
    @('G1_P012_MuJoCo_Windows_Package\deploy\deploy_mujoco\configs\g1_P1_6700.yaml','mujoco\configs\g1_P1_6700.yaml'),
    @('G1_P012_MuJoCo_Windows_Package\deploy\deploy_mujoco\configs\g1_P2_6700.yaml','mujoco\configs\g1_P2_6700.yaml'),
    @('G1_P012_MuJoCo_Windows_Package\README_Windows_MuJoCo_JA.md','mujoco\README_Windows_MuJoCo_JA.md'),
    @('G1_P012_MuJoCo_Windows_Package\checksums_and_validation.json','mujoco\metadata\checksums_and_validation.json'),
    @('upstream\unitree_rl_gym\LICENSE','mujoco\patches\UNITREE_RL_GYM_LICENSE.txt'),
    @('G1_P012_MuJoCo_Windows_Package\deploy\pre_train\g1\P0_6700_jit.pt','checkpoints\P0\P0_6700_jit.pt'),
    @('G1_P012_MuJoCo_Windows_Package\deploy\pre_train\g1\P1_6700_jit.pt','checkpoints\P1\P1_6700_jit.pt'),
    @('G1_P012_MuJoCo_Windows_Package\deploy\pre_train\g1\P2_6700_jit.pt','checkpoints\P2\P2_6700_jit.pt')
)
foreach ($row in $copyList) {
    $src = Join-Path $SourceRoot $row[0]
    $dst = Join-Path $RepoRoot $row[1]
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        Write-Warning "Not found (verify with Codex): $src"
        continue
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
    $hashSrc = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
    $hashDst = (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash
    if ($hashSrc -ne $hashDst) { throw "Hash mismatch: $src" }
    Write-Output "Copied and verified: $($row[1])"
}

# Adapt two text files to this public repository layout. These changes are
# intentionally made only in the destination after the byte-for-byte copy was
# verified above.
$recordPath = Join-Path $RepoRoot 'mujoco\scripts\record_g1.py'
$recordText = [IO.File]::ReadAllText($recordPath)
$recordOld = 'PROJECT_ROOT = Path(__file__).resolve().parents[1]'
$recordNew = "# Public repository layout: <repo>/mujoco/scripts/record_g1.py.`nPROJECT_ROOT = Path(__file__).resolve().parents[2]"
if (-not $recordText.Contains($recordOld)) {
    throw "Expected project-root line was not found: $recordPath"
}
[IO.File]::WriteAllText($recordPath, $recordText.Replace($recordOld, $recordNew), [Text.UTF8Encoding]::new($false))

$readmePath = Join-Path $RepoRoot 'mujoco\README_Windows_MuJoCo_JA.md'
$readmeText = [IO.File]::ReadAllText($readmePath)
$readmeText = [regex]::Replace(
    $readmeText,
    '(?i)[A-Z]:\\[^`\r\n]*?\\unitree_rl_gym',
    '<unitree_rl_gym>'
)
$publicNotice = "> **Public repository layout:** Follow [../README.md](../README.md) for the actual file placement. This document describes the original Windows package behavior.`n`n"
if (-not $readmeText.StartsWith($publicNotice)) {
    $readmeText = $publicNotice + $readmeText
}
[IO.File]::WriteAllText($readmePath, $readmeText, [Text.UTF8Encoding]::new($false))

Write-Output 'Applied public-layout adaptations to record_g1.py and README_Windows_MuJoCo_JA.md.'
Write-Output 'Done: known files copied. The deploy_mujoco patch must be generated and reviewed separately.'
