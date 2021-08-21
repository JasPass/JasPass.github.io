import { Directive, Input } from '@angular/core';

@Directive({
    selector: '[<%= prefix %><%= classify(name) %>]',
})
export class <%= classify(name) %>Directive {
    @Input() param!: string;

    constructor() {}
}
