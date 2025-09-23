pkgdatadir = get_option('prefix') / get_option('datadir') / meson.project_name()
bindir = get_option('prefix') / get_option('bindir')

python = find_program('python3', required: true)
sass = find_program('sassc', 'sass', required: true)

css = custom_target(
  'style.css',
  input: files('style.scss'),
  depend_files: files('timer/Timer.scss'),
  command: [sass, '@INPUT@', '@OUTPUT@'],
  output: 'style.css',
  build_by_default: true,
)

gnome = import('gnome')

resources = gnome.compile_resources(
  'resources',
  'gresource.xml',
  dependencies: [css],
  gresource_bundle: true,
  install: true,
  install_dir: pkgdatadir,
  extra_args: ['--sourcedir', meson.current_build_dir()],
)

python_packages = ['app', 'timer', 'aslrunner', 'asl_scripts', 'lssparser']
foreach pkg : python_packages
  install_subdir(
    pkg,
    install_dir: pkgdatadir,
    strip_directory: false,
    exclude_directories: ['__pycache__'],
  )
endforeach

install_data('main.py', install_dir: pkgdatadir)

launcher_conf = configuration_data()
launcher_conf.set('PYTHON', python.full_path())
launcher_conf.set('PKGDATADIR', pkgdatadir)

configure_file(
  input: 'astalsplit.in',
  output: meson.project_name(),
  configuration: launcher_conf,
  install: true,
  install_dir: bindir,
  install_mode: 'rwxr-xr-x',
)
