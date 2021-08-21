/* tslint:disable: ordered-imports*/
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';

/* Modules */
import { AppCommonModule } from '@common/app-common.module';
import { NavigationModule } from '@modules/navigation/navigation.module';

/* Components */
import * as testComponents from './components';

/* Containers */
import * as testContainers from './containers';

/* Guards */
import * as testGuards from './guards';

/* Services */
import * as testServices from './services';

@NgModule({
    imports: [
        CommonModule,
        RouterModule,
        ReactiveFormsModule,
        FormsModule,
        AppCommonModule,
        NavigationModule,
    ],
    providers: [...testServices.services, ...testGuards.guards],
    declarations: [...testContainers.containers, ...testComponents.components],
    exports: [...testContainers.containers, ...testComponents.components],
})
export class TestModule {}
